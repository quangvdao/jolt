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
            eprintln!("Warning: failed to create cache directory {:?}: {}", parent, e);
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

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sum-params-buffer"),
            size: std::mem::size_of::<Params>() as u64,
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

            let bind_group = self
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
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
                            resource: self
                                .scratch_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.params_buffer.as_entire_binding(),
                        },
                    ],
                });

            let mut encoder =
                self.device
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

            let bind_group = self
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sum-bind-group"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self
                                .q_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
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

            let mut encoder =
                self.device
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

        let mut encoder =
            self.device
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
}

fn main() {
    // Polynomial length exponent: if an argument is provided, interpret it as `log2_len`.
    // Example: `cargo run -p webgpu-sumcheck -- 22` => length = 2^22.
    let log2_len: u32 = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("Expected integer exponent for polynomial length"))
        .unwrap_or(20);
    // Keep this within a reasonable range to avoid exhausting memory on accident.
    assert!(
        log2_len <= 26,
        "Exponent too large for this demo (would allocate too much memory)"
    );
    let n: usize = 1usize << log2_len;

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
    let gpu_sum = gpu_ctx.sum_products_u128(&p, &q);
    let gpu_sum_time = start_gpu_sum.elapsed();

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
    println!("Time for GPU sum:              {:?}", gpu_sum_time);
    println!("Time for one binding step (CPU): {:?}", bind_time);
}
