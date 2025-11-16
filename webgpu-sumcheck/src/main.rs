use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;
use std::{fs, path::PathBuf};

const NUM_LIMBS: usize = 4; // 4 × 32-bit limbs = 128 bits
const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
struct U128Limbs {
    // Little-endian base-2^16 digits stored in u32 slots (lower 16 bits used)
    limbs: [u32; NUM_LIMBS],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    len: u32,
    phase: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SumcheckParams {
    len: u32,
    _pad0: u32,
    // Pad so that `r` starts at offset 16 to match WGSL's `vec4<u32>` alignment.
    _pad1: [u32; 2],
    r: U128Limbs,
}

fn u128_to_limbs(x: u128) -> U128Limbs {
    let mut tmp = x;
    let mut limbs = [0u32; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        limbs[i] = (tmp & 0xFFFF_FFFF) as u32;
        tmp >>= 32;
    }
    U128Limbs { limbs }
}

fn limbs_to_u128(l: &U128Limbs) -> u128 {
    let mut x: u128 = 0;
    // Most significant limb last
    for i in (0..NUM_LIMBS).rev() {
        x <<= 32;
        x |= (l.limbs[i]) as u128;
    }
    x
}

/// CPU reference: evaluate one sumcheck round for f = p * q on the current
/// evaluation tables for p and q, returning coefficients (g0, g1, g2) of
/// g(X) = g0 + g1 X + g2 X^2.
#[allow(dead_code)]
fn cpu_sumcheck_eval_round(p: &[u128], q: &[u128]) -> (u128, u128, u128) {
    assert_eq!(
        p.len(),
        q.len(),
        "cpu_sumcheck_eval_round: p,q length mismatch"
    );
    assert!(
        p.len() % 2 == 0,
        "cpu_sumcheck_eval_round: length must be even"
    );
    let mut g0 = 0u128;
    let mut g1 = 0u128;
    let mut g2 = 0u128;
    let pairs = p.len() / 2;
    for i in 0..pairs {
        let p0 = p[2 * i];
        let p1 = p[2 * i + 1];
        let q0 = q[2 * i];
        let q1 = q[2 * i + 1];
        let dp = p1.wrapping_sub(p0);
        let dq = q1.wrapping_sub(q0);

        let a = p0.wrapping_mul(q0);
        let b1 = p0.wrapping_mul(dq);
        let b2 = q0.wrapping_mul(dp);
        let b = b1.wrapping_add(b2);
        let c = dp.wrapping_mul(dq);

        g0 = g0.wrapping_add(a);
        g1 = g1.wrapping_add(b);
        g2 = g2.wrapping_add(c);
    }
    (g0, g1, g2)
}

/// CPU reference: bind one variable with challenge `r` into the evaluation
/// tables for p and q in place. The first half of each table is updated with
/// p_next[i] = p0 + r * (p1 - p0), q_next[i] = q0 + r * (q1 - q0), where
/// p0 = p[2*i], p1 = p[2*i+1], etc. The length is halved by the caller.
#[allow(dead_code)]
fn cpu_sumcheck_apply_challenge(p: &mut [u128], q: &mut [u128], r: u128) {
    assert_eq!(
        p.len(),
        q.len(),
        "cpu_sumcheck_apply_challenge: p,q length mismatch"
    );
    assert!(
        p.len() % 2 == 0,
        "cpu_sumcheck_apply_challenge: length must be even"
    );
    let pairs = p.len() / 2;
    for i in 0..pairs {
        let p0 = p[2 * i];
        let p1 = p[2 * i + 1];
        let q0 = q[2 * i];
        let q1 = q[2 * i + 1];
        let dp = p1.wrapping_sub(p0);
        let dq = q1.wrapping_sub(q0);
        let rp = r.wrapping_mul(dp);
        let rq = r.wrapping_mul(dq);
        p[i] = p0.wrapping_add(rp);
        q[i] = q0.wrapping_add(rq);
    }
}

/// Basic binding step for a multilinear polynomial with u128 coefficients.
/// Treats `poly` as evaluations over {0,1}^n ordered so that the bound variable
/// is the least-significant index bit. Returns a new polynomial of half length.
fn bind_poly_var_bot_u128(poly: &[u128], r: u128) -> Vec<u128> {
    assert!(
        poly.len() % 2 == 0,
        "poly length must be even for a single binding step"
    );
    let n = poly.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let low = poly[2 * i];
        let high = poly[2 * i + 1];
        let m = high.wrapping_sub(low);
        out.push(low.wrapping_add(r.wrapping_mul(m)));
    }
    out
}

fn cache_file_path(log2_len: u32) -> PathBuf {
    // Place cached polynomials under `<crate>/data/pq_2exp{log2_len}.bin`
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    base.join("data").join(format!("pq_2exp{log2_len}.bin"))
}

fn try_load_cached_polys(
    log2_len: u32,
    n: usize,
) -> Option<(Vec<u128>, Vec<u128>, std::time::Duration)> {
    let path = cache_file_path(log2_len);
    let start = Instant::now();
    let data = fs::read(&path).ok()?;

    let expected_elems = 2usize * n;
    let expected_bytes = expected_elems
        .checked_mul(std::mem::size_of::<u128>())
        .unwrap();
    if data.len() != expected_bytes {
        return None;
    }

    let mut p = Vec::with_capacity(n);
    let mut q = Vec::with_capacity(n);

    // Layout: [p[0..n), q[0..n)] as little-endian u128
    for i in 0..n {
        let offset = i * 16;
        let mut buf = [0u8; 16];
        buf.copy_from_slice(&data[offset..offset + 16]);
        p.push(u128::from_le_bytes(buf));
    }
    for i in 0..n {
        let offset = (n + i) * 16;
        let mut buf = [0u8; 16];
        buf.copy_from_slice(&data[offset..offset + 16]);
        q.push(u128::from_le_bytes(buf));
    }

    let load_time = start.elapsed();
    Some((p, q, load_time))
}

fn save_cached_polys(log2_len: u32, p: &[u128], q: &[u128]) {
    let path = cache_file_path(log2_len);
    if let Some(parent) = path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            eprintln!(
                "Warning: failed to create cache directory {:?}: {}",
                parent, e
            );
            return;
        }
    }

    let mut bytes = Vec::with_capacity((p.len() + q.len()) * 16);
    for &x in p {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    for &x in q {
        bytes.extend_from_slice(&x.to_le_bytes());
    }

    if let Err(e) = fs::write(&path, &bytes) {
        eprintln!("Warning: failed to write cache file {:?}: {}", path, e);
    }
}

const SHADER_SRC: &str = include_str!("../shaders/u128_mul_sum.wgsl");

struct GpuSumContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
    params_buffer: wgpu::Buffer,
    // Sumcheck-specific pipeline and layout (uses @group(1) in the shader)
    sumcheck_bind_group_layout: wgpu::BindGroupLayout,
    sumcheck_eval_pipeline: wgpu::ComputePipeline,
    sumcheck_bind_pipeline: wgpu::ComputePipeline,
    sumcheck_reduce_pipeline: wgpu::ComputePipeline,
    sumcheck_params_buffer: wgpu::Buffer,
    p_buffer: Option<wgpu::Buffer>,
    q_buffer: Option<wgpu::Buffer>,
    scratch_buffer: Option<wgpu::Buffer>,
    buffer_capacity: usize, // number of U128Limbs elements
    max_chunk_elems: usize, // maximum elements per GPU pass (binding size limit)
}

impl GpuSumContext {
    async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapter found");

        let info = adapter.get_info();
        println!(
            "Using adapter: {} (backend: {:?}, device_type: {:?})",
            info.name, info.backend, info.device_type
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("webgpu-sumcheck-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("u128-sum-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sum-bind-group-layout"),
            entries: &[
                // p buffer (input)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // q buffer (only used in the first phase)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // out buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // params
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sum-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("u128-sum-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // Sumcheck bind group layout (corresponds to @group(1) bindings in WGSL)
        let sumcheck_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sumcheck-bind-group-layout"),
                entries: &[
                    // sc_p_in
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // sc_p_out
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // sc_q_in
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // sc_q_out
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // sc_g0_partial
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // sc_g1_partial
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // sc_g2_partial
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // sc_params
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let sumcheck_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sumcheck-pipeline-layout"),
                // Sumcheck kernels only use @group(1) in the WGSL, so we attach
                // a single bind group layout here and bind it at index 0.
                bind_group_layouts: &[&sumcheck_bind_group_layout],
                push_constant_ranges: &[],
            });

        let sumcheck_eval_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sumcheck-eval-pipeline"),
                layout: Some(&sumcheck_pipeline_layout),
                module: &shader,
                entry_point: "sumcheck_eval_round",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let sumcheck_bind_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sumcheck-bind-pipeline"),
                layout: Some(&sumcheck_pipeline_layout),
                module: &shader,
                entry_point: "sumcheck_bind_round",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let sumcheck_reduce_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sumcheck-reduce-pipeline"),
                layout: Some(&sumcheck_pipeline_layout),
                module: &shader,
                entry_point: "sumcheck_reduce_coeffs",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sum-params-buffer"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sumcheck_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sumcheck-params-buffer"),
            size: std::mem::size_of::<SumcheckParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Compute the maximum number of U128Limbs we can bind in a single storage buffer
        // based on the device's binding size limit.
        let device_limits = device.limits();
        let element_size = std::mem::size_of::<U128Limbs>() as u64;
        let max_chunk_elems =
            (device_limits.max_storage_buffer_binding_size as u64 / element_size) as usize;

        Self {
            device,
            queue,
            bind_group_layout,
            compute_pipeline,
            params_buffer,
            sumcheck_bind_group_layout,
            sumcheck_eval_pipeline,
            sumcheck_bind_pipeline,
            sumcheck_reduce_pipeline,
            sumcheck_params_buffer,
            p_buffer: None,
            q_buffer: None,
            scratch_buffer: None,
            buffer_capacity: 0,
            max_chunk_elems,
        }
    }

    fn ensure_buffers(&mut self, required_elems: usize) {
        if required_elems == 0 {
            return;
        }

        debug_assert!(
            required_elems <= self.max_chunk_elems,
            "Requested chunk size {} exceeds max_chunk_elems {}",
            required_elems,
            self.max_chunk_elems
        );

        if self.buffer_capacity >= required_elems
            && self.p_buffer.is_some()
            && self.q_buffer.is_some()
            && self.scratch_buffer.is_some()
        {
            return;
        }

        let element_size = std::mem::size_of::<U128Limbs>() as u64;
        let buffer_size_bytes = (required_elems as u64) * element_size;

        let p_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sum-p-buffer"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let q_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sum-q-buffer"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scratch_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sum-scratch-buffer"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.p_buffer = Some(p_buffer);
        self.q_buffer = Some(q_buffer);
        self.scratch_buffer = Some(scratch_buffer);
        self.buffer_capacity = required_elems;
    }

    fn sum_products_chunk_u128(&mut self, p: &[u128], q: &[u128]) -> u128 {
        if p.is_empty() {
            return 0;
        }
        assert_eq!(p.len(), q.len(), "p and q must have the same length");
        assert!(
            p.len() <= self.max_chunk_elems,
            "Chunk length {} exceeds max_chunk_elems {}",
            p.len(),
            self.max_chunk_elems
        );

        // Encode inputs as limb representation
        let encoded_p: Vec<U128Limbs> = p.iter().copied().map(u128_to_limbs).collect();
        let encoded_q: Vec<U128Limbs> = q.iter().copied().map(u128_to_limbs).collect();
        let len = encoded_p.len();

        self.ensure_buffers(len);

        let element_size = std::mem::size_of::<U128Limbs>() as u64;

        // Write input data into the p and q buffers
        let p_buf = self.p_buffer.as_ref().expect("p buffer missing");
        let q_buf = self.q_buffer.as_ref().expect("q buffer missing");
        self.queue
            .write_buffer(p_buf, 0, bytemuck::cast_slice(&encoded_p));
        self.queue
            .write_buffer(q_buf, 0, bytemuck::cast_slice(&encoded_q));

        let mut current_len = len as u32;
        let mut use_p_as_input = true; // track which buffer currently holds the input

        // First pass: compute p[i] * q[i] and reduce within workgroups.
        if current_len > 1 {
            let workgroup_count =
                ((current_len as u64) + (WORKGROUP_SIZE as u64) - 1) / (WORKGROUP_SIZE as u64);
            let workgroup_count_u32 = workgroup_count as u32;

            let params = Params {
                len: current_len,
                phase: 0,
                _pad1: 0,
                _pad2: 0,
            };

            self.queue
                .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sum-bind-group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: p_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: q_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.scratch_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sum-command-encoder"),
                });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sum-compute-pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.compute_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(workgroup_count_u32, 1, 1);
            }

            self.queue.submit(Some(encoder.finish()));

            current_len = workgroup_count_u32;
            use_p_as_input = false;
        }

        // Subsequent passes: sum partial results only (no further multiplication).
        while current_len > 1 {
            let workgroup_count =
                ((current_len as u64) + (WORKGROUP_SIZE as u64) - 1) / (WORKGROUP_SIZE as u64);
            let workgroup_count_u32 = workgroup_count as u32;

            let params = Params {
                len: current_len,
                phase: 1,
                _pad1: 0,
                _pad2: 0,
            };

            self.queue
                .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

            let (input_buf, output_buf) = if use_p_as_input {
                (
                    self.p_buffer.as_ref().unwrap(),
                    self.scratch_buffer.as_ref().unwrap(),
                )
            } else {
                (
                    self.scratch_buffer.as_ref().unwrap(),
                    self.p_buffer.as_ref().unwrap(),
                )
            };

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sum-bind-group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.q_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sum-command-encoder"),
                });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sum-compute-pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.compute_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(workgroup_count_u32, 1, 1);
            }

            self.queue.submit(Some(encoder.finish()));

            current_len = workgroup_count_u32;
            use_p_as_input = !use_p_as_input;
        }

        // Read back the single remaining element. The final value resides in whichever
        // buffer was last used as the output buffer.
        let final_buf = if use_p_as_input {
            self.p_buffer.as_ref().unwrap()
        } else {
            self.scratch_buffer.as_ref().unwrap()
        };

        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sum-result-buffer"),
            size: element_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sum-readback-encoder"),
            });
        encoder.copy_buffer_to_buffer(final_buf, 0, &result_buffer, 0, element_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = result_buffer.slice(..);

        // wgpu 0.20 uses callback-based mapping; block on completion using a channel.
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).ok();
        });
        // Wait for the GPU to finish the mapping operation.
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("Failed to receive map_async result")
            .expect("Failed to map result buffer");
        let data = buffer_slice.get_mapped_range();

        let result_limbs: U128Limbs =
            *bytemuck::from_bytes::<U128Limbs>(&data[..std::mem::size_of::<U128Limbs>()]);

        drop(data);
        result_buffer.unmap();

        limbs_to_u128(&result_limbs)
    }

    fn sum_products_u128(&mut self, p: &[u128], q: &[u128]) -> u128 {
        if p.is_empty() {
            return 0;
        }
        assert_eq!(p.len(), q.len(), "p and q must have the same length");

        let mut total: u128 = 0;
        let mut offset = 0usize;
        let len = p.len();

        while offset < len {
            let remaining = len - offset;
            let chunk_len = remaining.min(self.max_chunk_elems);
            let chunk_p = &p[offset..offset + chunk_len];
            let chunk_q = &q[offset..offset + chunk_len];
            let chunk_sum = self.sum_products_chunk_u128(chunk_p, chunk_q);
            total = total.wrapping_add(chunk_sum);
            offset += chunk_len;
        }

        total
    }

    /// Debug helper: read back `len` U128Limbs elements from a GPU buffer and
    /// decode them into u128 values. Intended for small instances to compare
    /// GPU state against CPU reference implementations.
    fn debug_read_u128_vector(&self, buffer: &wgpu::Buffer, len: u32) -> Vec<u128> {
        if len == 0 {
            return Vec::new();
        }
        let element_size = std::mem::size_of::<U128Limbs>() as u64;
        let size_bytes = (len as u64) * element_size;

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug-readback-buffer"),
            size: size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("debug-readback-encoder"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, size_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..size_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("Failed to receive map_async result for debug_read_u128_vector")
            .expect("Failed to map debug_readback buffer");

        let data = slice.get_mapped_range();
        let limbs_slice: &[U128Limbs] = bytemuck::cast_slice(&data);
        let mut out = Vec::with_capacity(len as usize);
        for i in 0..(len as usize) {
            out.push(limbs_to_u128(&limbs_slice[i]));
        }

        drop(data);
        readback.unmap();

        out
    }
}

/// State for a GPU-backed sumcheck for f = p * q, where p and q are multilinear
/// in each variable. We track evaluation tables for p and q separately.
struct GpuSumcheckState {
    p_in: wgpu::Buffer,
    p_out: wgpu::Buffer,
    q_in: wgpu::Buffer,
    q_out: wgpu::Buffer,
    g0_partial: wgpu::Buffer,
    g1_partial: wgpu::Buffer,
    g2_partial: wgpu::Buffer,
    current_len: u32,
    // log2_len: u32,
}

impl GpuSumContext {
    /// Initialize a sumcheck state on the GPU with the given evaluation tables
    /// for p and q. Both must have the same length, which must be a power of two.
    fn sumcheck_start(&self, p: &[u128], q: &[u128]) -> GpuSumcheckState {
        assert!(
            !p.is_empty(),
            "sumcheck_start requires a non-empty evaluation vector"
        );
        assert_eq!(
            p.len(),
            q.len(),
            "sumcheck_start requires p and q to have the same length"
        );
        let len = p.len();
        assert_eq!(
            len & (len - 1),
            0,
            "sumcheck_start expects length to be a power of two"
        );

        let encoded_p: Vec<U128Limbs> = p.iter().copied().map(u128_to_limbs).collect();
        let encoded_q: Vec<U128Limbs> = q.iter().copied().map(u128_to_limbs).collect();

        let element_size = std::mem::size_of::<U128Limbs>() as u64;
        let buffer_size_bytes = (len as u64) * element_size;

        let p_in = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sumcheck-p-in"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let p_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sumcheck-p-out"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let q_in = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sumcheck-q-in"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let q_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sumcheck-q-out"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Maximum number of pairs occurs in the first round.
        let max_pairs = len / 2;
        let max_workgroups =
            ((max_pairs as u64) + (WORKGROUP_SIZE as u64) - 1) / (WORKGROUP_SIZE as u64);
        let partial_len = max_workgroups as usize;
        let partial_buffer_size = (partial_len as u64) * element_size;

        let g0_partial = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sumcheck-g0-partial"),
            size: partial_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let g1_partial = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sumcheck-g1-partial"),
            size: partial_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let g2_partial = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sumcheck-g2-partial"),
            size: partial_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Upload initial values into p_in and q_in.
        self.queue
            .write_buffer(&p_in, 0, bytemuck::cast_slice(&encoded_p));
        self.queue
            .write_buffer(&q_in, 0, bytemuck::cast_slice(&encoded_q));

        GpuSumcheckState {
            p_in,
            p_out,
            q_in,
            q_out,
            g0_partial,
            g1_partial,
            g2_partial,
            current_len: len as u32,
            // log2_len,
        }
    }

    /// Compute the degree-2 univariate g(X) = g0 + g1 X + g2 X^2 for the current
    /// sumcheck round over f = p * q. This only reads the current evaluation
    /// tables and does not apply a challenge.
    fn sumcheck_eval_round(&self, state: &GpuSumcheckState) -> (u128, u128, u128) {
        assert!(
            state.current_len >= 2,
            "sumcheck_eval_round requires at least one pair"
        );
        assert_eq!(
            state.current_len % 2,
            0,
            "current_len must be even in sumcheck_eval_round"
        );

        let len = state.current_len;
        let pairs = len / 2;
        let workgroup_count =
            ((pairs as u64) + (WORKGROUP_SIZE as u64) - 1) / (WORKGROUP_SIZE as u64);
        let workgroup_count_u32 = workgroup_count as u32;

        let params = SumcheckParams {
            len,
            _pad0: 0,
            _pad1: [0u32; 2],
            r: U128Limbs {
                limbs: [0u32; NUM_LIMBS],
            },
        };

        self.queue
            .write_buffer(&self.sumcheck_params_buffer, 0, bytemuck::bytes_of(&params));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sumcheck-eval-bind-group"),
            layout: &self.sumcheck_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.p_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state.p_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state.q_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state.q_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: state.g0_partial.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: state.g1_partial.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: state.g2_partial.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.sumcheck_params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sumcheck-eval-encoder"),
            });

        // First pass: compute per-workgroup partial sums of A, B, C.
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sumcheck-eval-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.sumcheck_eval_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroup_count_u32, 1, 1);
        }

        // Subsequent passes: reduce the partial sums on GPU until one triple remains.
        let mut partial_len = workgroup_count_u32;
        while partial_len > 1 {
            let reduce_workgroups =
                ((partial_len as u64) + (WORKGROUP_SIZE as u64) - 1) / (WORKGROUP_SIZE as u64);
            let reduce_workgroups_u32 = reduce_workgroups as u32;

            let reduce_params = SumcheckParams {
                len: partial_len,
                _pad0: 0,
                _pad1: [0u32; 2],
                r: U128Limbs {
                    limbs: [0u32; NUM_LIMBS],
                },
            };

            self.queue.write_buffer(
                &self.sumcheck_params_buffer,
                0,
                bytemuck::bytes_of(&reduce_params),
            );

            let reduce_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sumcheck-reduce-bind-group"),
                layout: &self.sumcheck_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: state.p_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: state.p_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: state.q_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: state.q_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: state.g0_partial.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: state.g1_partial.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: state.g2_partial.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.sumcheck_params_buffer.as_entire_binding(),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sumcheck-reduce-pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.sumcheck_reduce_pipeline);
                cpass.set_bind_group(0, &reduce_bind_group, &[]);
                cpass.dispatch_workgroups(reduce_workgroups_u32, 1, 1);
            }

            partial_len = reduce_workgroups_u32;
        }

        // Read back the single remaining triple (three U128Limbs).
        let element_size = std::mem::size_of::<U128Limbs>() as u64;

        let g0_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sumcheck-g0-readback"),
            size: element_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let g1_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sumcheck-g1-readback"),
            size: element_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let g2_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sumcheck-g2-readback"),
            size: element_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&state.g0_partial, 0, &g0_readback, 0, element_size);
        encoder.copy_buffer_to_buffer(&state.g1_partial, 0, &g1_readback, 0, element_size);
        encoder.copy_buffer_to_buffer(&state.g2_partial, 0, &g2_readback, 0, element_size);

        self.queue.submit(Some(encoder.finish()));

        // Map and decode the three coefficients.
        let g0_slice = g0_readback.slice(..element_size);
        let (g0_tx, g0_rx) = std::sync::mpsc::channel();
        g0_slice.map_async(wgpu::MapMode::Read, move |res| {
            g0_tx.send(res).ok();
        });

        let g1_slice = g1_readback.slice(..element_size);
        let (g1_tx, g1_rx) = std::sync::mpsc::channel();
        g1_slice.map_async(wgpu::MapMode::Read, move |res| {
            g1_tx.send(res).ok();
        });

        let g2_slice = g2_readback.slice(..element_size);
        let (g2_tx, g2_rx) = std::sync::mpsc::channel();
        g2_slice.map_async(wgpu::MapMode::Read, move |res| {
            g2_tx.send(res).ok();
        });

        self.device.poll(wgpu::Maintain::Wait);

        g0_rx
            .recv()
            .expect("Failed to receive map_async result for g0")
            .expect("Failed to map g0 buffer");
        g1_rx
            .recv()
            .expect("Failed to receive map_async result for g1")
            .expect("Failed to map g1 buffer");
        g2_rx
            .recv()
            .expect("Failed to receive map_async result for g2")
            .expect("Failed to map g2 buffer");

        let g0_data = g0_slice.get_mapped_range();
        let g1_data = g1_slice.get_mapped_range();
        let g2_data = g2_slice.get_mapped_range();

        let g0_limbs: &U128Limbs =
            bytemuck::from_bytes(&g0_data[..std::mem::size_of::<U128Limbs>()]);
        let g1_limbs: &U128Limbs =
            bytemuck::from_bytes(&g1_data[..std::mem::size_of::<U128Limbs>()]);
        let g2_limbs: &U128Limbs =
            bytemuck::from_bytes(&g2_data[..std::mem::size_of::<U128Limbs>()]);

        let g0 = limbs_to_u128(g0_limbs);
        let g1 = limbs_to_u128(g1_limbs);
        let g2 = limbs_to_u128(g2_limbs);

        drop(g0_data);
        drop(g1_data);
        drop(g2_data);
        g0_readback.unmap();
        g1_readback.unmap();
        g2_readback.unmap();

        (g0, g1, g2)
    }

    /// Apply a verifier challenge r to advance the sumcheck state:
    ///   p_out[i] = p0 + r * (p1 - p0)
    ///   q_out[i] = q0 + r * (q1 - q0)
    /// and swap `p_in` / `p_out` and `q_in` / `q_out`, halving `current_len`.
    fn sumcheck_apply_challenge(&self, state: &mut GpuSumcheckState, r: u128) {
        assert!(
            state.current_len >= 2,
            "sumcheck_apply_challenge requires at least one pair"
        );
        assert_eq!(
            state.current_len % 2,
            0,
            "current_len must be even in sumcheck_apply_challenge"
        );

        let len = state.current_len;
        let pairs = len / 2;
        let workgroup_count =
            ((pairs as u64) + (WORKGROUP_SIZE as u64) - 1) / (WORKGROUP_SIZE as u64);
        let workgroup_count_u32 = workgroup_count as u32;

        let r_limbs = u128_to_limbs(r);
        let params = SumcheckParams {
            len,
            _pad0: 0,
            _pad1: [0u32; 2],
            r: r_limbs,
        };

        self.queue
            .write_buffer(&self.sumcheck_params_buffer, 0, bytemuck::bytes_of(&params));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sumcheck-bind-bind-group"),
            layout: &self.sumcheck_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.p_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state.p_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state.q_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state.q_out.as_entire_binding(),
                },
                // partial buffers not used by this kernel, but must be bound
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: state.g0_partial.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: state.g1_partial.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: state.g2_partial.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.sumcheck_params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sumcheck-bind-encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sumcheck-bind-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.sumcheck_bind_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroup_count_u32, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        // Swap input/output buffers and halve the current length for p and q.
        std::mem::swap(&mut state.p_in, &mut state.p_out);
        std::mem::swap(&mut state.q_in, &mut state.q_out);
        state.current_len = pairs;
    }
}

fn main() {
    // CLI:
    //   cargo run -p webgpu-sumcheck -- <log2_len> [gpu_iters]
    // where gpu_iters is the number of times to run the GPU kernel for profiling.
    let mut args = std::env::args();
    args.next(); // program name

    // Polynomial length exponent: if an argument is provided, interpret it as `log2_len`.
    // Example: `cargo run -p webgpu-sumcheck -- 22` => length = 2^22.
    let log2_len: u32 = args
        .next()
        .map(|s| {
            s.parse()
                .expect("Expected integer exponent for polynomial length")
        })
        .unwrap_or(20);

    let iterations: u32 = args
        .next()
        .map(|s| s.parse().expect("Expected integer iteration count"))
        .unwrap_or(1);
    // Keep this within a reasonable range to avoid exhausting memory on accident.
    assert!(
        log2_len <= 26,
        "Exponent too large for this demo (would allocate too much memory)"
    );
    let n: usize = 1usize << log2_len;

    // Pre-sample verifier challenges for the sumcheck rounds so that both the
    // CPU reference and GPU implementations use the same sequence.
    let mut challenge_rng = rand::thread_rng();
    let challenges: Vec<u128> = (0..log2_len).map(|_| challenge_rng.gen::<u128>()).collect();

    // Try to load cached polynomials for this size; if missing or invalid, generate and cache.
    let (p, q, rng_time, load_time, used_cache) =
        if let Some((p, q, load_time)) = try_load_cached_polys(log2_len, n) {
            (p, q, std::time::Duration::ZERO, load_time, true)
        } else {
            let start_rng = Instant::now();
            let mut rng = rand::thread_rng();
            let p: Vec<u128> = (0..n).map(|_| rng.gen::<u128>()).collect();
            let q: Vec<u128> = (0..n).map(|_| rng.gen::<u128>()).collect();
            let rng_time = start_rng.elapsed();
            save_cached_polys(log2_len, &p, &q);
            (p, q, rng_time, std::time::Duration::ZERO, false)
        };

    let start_cpu = Instant::now();
    // Compute coefficient-wise products on CPU using true u128 arithmetic
    let products: Vec<u128> = p
        .iter()
        .zip(q.iter())
        .map(|(a, b)| a.wrapping_mul(*b))
        .collect();

    let cpu_sum: u128 = products
        .iter()
        .copied()
        .fold(0u128, |acc, x| acc.wrapping_add(x));
    let cpu_time = start_cpu.elapsed();

    // Multi-threaded CPU sum over the same products using Rayon
    let start_cpu_mt = Instant::now();
    let cpu_sum_mt: u128 = products
        .par_iter()
        .copied()
        .reduce(|| 0u128, |acc, x| acc.wrapping_add(x));
    let cpu_mt_time = start_cpu_mt.elapsed();

    debug_assert_eq!(cpu_sum, cpu_sum_mt);

    // Initialize GPU context once (device, pipeline, etc.).
    let start_gpu_setup = Instant::now();
    let mut gpu_ctx = pollster::block_on(GpuSumContext::new());
    let gpu_setup_time = start_gpu_setup.elapsed();

    let start_gpu_sum = Instant::now();
    let mut gpu_sum = 0u128;
    for _ in 0..iterations {
        gpu_sum = gpu_ctx.sum_products_u128(&p, &q);
    }
    let gpu_sum_time = start_gpu_sum.elapsed();
    let gpu_sum_time_per_iter = gpu_sum_time / iterations;

    // Optional CPU reference sumcheck to sanity-check the algebraic recurrence
    // and compare GPU state against CPU state round-by-round. Only enabled for
    // small instances to keep debug output manageable.
    if n <= 32 {
        let mut p_cpu = p.clone();
        let mut q_cpu = q.clone();
        let mut len_cpu = n;
        let mut prev_g_cpu: Option<(u128, u128, u128, u128)> = None;
        let mut cpu_sumcheck_ok = true;

        // Separate GPU state used only for this debug comparison.
        let mut sc_state_debug = gpu_ctx.sumcheck_start(&p, &q);
        let mut gpu_state_ok = true;

        for round in 0..log2_len {
            let slice_len = len_cpu;
            let (g0_cpu, g1_cpu, g2_cpu) =
                cpu_sumcheck_eval_round(&p_cpu[..slice_len], &q_cpu[..slice_len]);

            let g_at_0_cpu = g0_cpu;
            let g_at_1_cpu = g0_cpu.wrapping_add(g1_cpu).wrapping_add(g2_cpu);

            let (g0_gpu, g1_gpu, g2_gpu) = gpu_ctx.sumcheck_eval_round(&sc_state_debug);

            if g0_cpu != g0_gpu || g1_cpu != g1_gpu || g2_cpu != g2_gpu {
                println!(
                    "[DEBUG] round {}: CPU g = ({:#034x}, {:#034x}, {:#034x}), \
                     GPU g = ({:#034x}, {:#034x}, {:#034x})",
                    round, g0_cpu, g1_cpu, g2_cpu, g0_gpu, g1_gpu, g2_gpu
                );
                cpu_sumcheck_ok = false;
                gpu_state_ok = false;
                break;
            }

            if round == 0 {
                let total = g_at_0_cpu.wrapping_add(g_at_1_cpu);
                if total != cpu_sum {
                    cpu_sumcheck_ok = false;
                }
                println!(
                    "[CPU] round 0: g0={:#034x}, g1={:#034x}, g2={:#034x}",
                    g0_cpu, g1_cpu, g2_cpu
                );
                println!(
                    "[CPU] round 0: g(0)+g(1)={:#034x}, cpu_sum={:#034x}",
                    total, cpu_sum
                );
            } else if let Some((prev_g0, prev_g1, prev_g2, r_prev)) = prev_g_cpu {
                let r_prev_sq = r_prev.wrapping_mul(r_prev);
                let prev_val = prev_g0
                    .wrapping_add(r_prev.wrapping_mul(prev_g1))
                    .wrapping_add(r_prev_sq.wrapping_mul(prev_g2));
                let total = g_at_0_cpu.wrapping_add(g_at_1_cpu);
                if total != prev_val {
                    cpu_sumcheck_ok = false;
                }
                println!(
                    "[CPU] round {}: g0={:#034x}, g1={:#034x}, g2={:#034x}",
                    round, g0_cpu, g1_cpu, g2_cpu
                );
                println!(
                    "[CPU] round {}: g(0)+g(1)={:#034x}, g_prev(r_prev)={:#034x}",
                    round, total, prev_val
                );
            }

            let r_j: u128 = challenges[round as usize];
            prev_g_cpu = Some((g0_cpu, g1_cpu, g2_cpu, r_j));

            // Apply the same challenge on CPU and GPU.
            cpu_sumcheck_apply_challenge(&mut p_cpu[..slice_len], &mut q_cpu[..slice_len], r_j);
            len_cpu /= 2;

            gpu_ctx.sumcheck_apply_challenge(&mut sc_state_debug, r_j);

            // Compare full evaluation tables after binding.
            let p_gpu = gpu_ctx.debug_read_u128_vector(&sc_state_debug.p_in, len_cpu as u32);
            let q_gpu = gpu_ctx.debug_read_u128_vector(&sc_state_debug.q_in, len_cpu as u32);
            for i in 0..len_cpu {
                if p_cpu[i] != p_gpu[i] || q_cpu[i] != q_gpu[i] {
                    println!(
                        "[DEBUG] round {}: mismatch at index {}: \
                         p_cpu={:#034x}, p_gpu={:#034x}, \
                         q_cpu={:#034x}, q_gpu={:#034x}",
                        round, i, p_cpu[i], p_gpu[i], q_cpu[i], q_gpu[i],
                    );
                    gpu_state_ok = false;
                    break;
                }
            }
            if !gpu_state_ok {
                break;
            }
        }

        println!("[CPU] sumcheck consistency: {}", cpu_sumcheck_ok);
        println!(
            "[DEBUG] GPU state matches CPU after binding: {}",
            gpu_state_ok
        );
    }

    // Run a GPU-backed sumcheck for f = p * q, where p and q are multilinear
    // and f has degree 2 in each variable. The prover messages are degree-2
    // univariates g_j(X) = g0 + g1 X + g2 X^2.
    let start_sumcheck = Instant::now();
    let mut sc_state = gpu_ctx.sumcheck_start(&p, &q);
    let mut prev_g: Option<(u128, u128, u128, u128)> = None;
    let mut sumcheck_ok = true;

    for round in 0..log2_len {
        let (g0, g1, g2) = gpu_ctx.sumcheck_eval_round(&sc_state);

        // Evaluate g_j at 0 and 1.
        let g_at_0 = g0;
        let g_at_1 = g0.wrapping_add(g1).wrapping_add(g2); // 1^2 = 1

        if round == 0 {
            // First round: g_0(0) + g_0(1) must equal the claimed sum S.
            let total = g_at_0.wrapping_add(g_at_1);
            if total != cpu_sum {
                sumcheck_ok = false;
            }
            println!(
                "sumcheck round 0: g0={:#034x}, g1={:#034x}, g2={:#034x}",
                g0, g1, g2
            );
            println!(
                "sumcheck round 0: g(0)+g(1)={:#034x}, cpu_sum={:#034x}",
                total, cpu_sum
            );
            debug_assert_eq!(total, cpu_sum, "sumcheck round 0: g(0)+g(1) != claimed sum");
        } else if let Some((prev_g0, prev_g1, prev_g2, r_prev)) = prev_g {
            // Later rounds: g_j(0) + g_j(1) must equal g_{j-1}(r_{j-1}).
            let r_prev_sq = r_prev.wrapping_mul(r_prev);
            let prev_val = prev_g0
                .wrapping_add(r_prev.wrapping_mul(prev_g1))
                .wrapping_add(r_prev_sq.wrapping_mul(prev_g2));
            let total = g_at_0.wrapping_add(g_at_1);
            if total != prev_val {
                sumcheck_ok = false;
            }
            println!(
                "sumcheck round {}: g(0)+g(1)={:#034x}, g_prev(r_prev)={:#034x}",
                round, total, prev_val
            );
            debug_assert_eq!(
                total, prev_val,
                "sumcheck round {}: g(0)+g(1) != g_prev(r_prev)",
                round
            );
        }

        // Sample verifier challenge r_j and bind p_j, q_j → p_{j+1}, q_{j+1} on GPU.
        let r_j: u128 = challenges[round as usize];
        prev_g = Some((g0, g1, g2, r_j));
        gpu_ctx.sumcheck_apply_challenge(&mut sc_state, r_j);
    }

    let sumcheck_time = start_sumcheck.elapsed();

    // Demonstrate a single binding step on the same polynomial using the standard
    // low/high formula from DensePolynomial::bound_poly_var_bot.
    let r_bind: u128 = rand::thread_rng().gen();
    let start_bind = Instant::now();
    let bound = bind_poly_var_bot_u128(&products, r_bind);
    let bind_time = start_bind.elapsed();

    println!("Polynomial log2 length: {}", log2_len);
    println!("Polynomial length: {}", n);
    println!("Bound polynomial length (after 1 var): {}", bound.len());
    println!("Used cached polys: {}", used_cache);
    println!("CPU sum of p[i] * q[i]:  {:#034x}", cpu_sum);
    println!("CPU MT sum of p[i] * q[i]:{:#034x}", cpu_sum_mt);
    println!("GPU sum of p[i] * q[i]:  {:#034x}", gpu_sum);
    println!("Match: {}", cpu_sum == gpu_sum);
    if used_cache {
        println!("Time to load cached polys:    {:?}", load_time);
    } else {
        println!("Time to generate random polys: {:?}", rng_time);
    }
    println!("Time for CPU products + sum:   {:?}", cpu_time);
    println!("Time for CPU MT sum (Rayon):   {:?}", cpu_mt_time);
    println!("Time for GPU setup (context):  {:?}", gpu_setup_time);
    println!("GPU iterations:                {}", iterations);
    println!("Time for GPU sum (total):      {:?}", gpu_sum_time);
    println!("Time for GPU sum per iter:     {:?}", gpu_sum_time_per_iter);
    println!("GPU sumcheck consistency:      {}", sumcheck_ok);
    println!("Time for GPU sumcheck:         {:?}", sumcheck_time);
    println!("Time for one binding step (CPU): {:?}", bind_time);
}
