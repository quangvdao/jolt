use std::time::Instant;

use ark_bn254::Fr;
use ark_ff::PrimeField;
use rand::Rng;
use rayon::prelude::*;

const FR_NUM_LIMBS: usize = 8; // 8 × 32-bit limbs = 256 bits for BN254 Fr
const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
struct FrLimbs {
    // Little-endian base-2^32 digits stored in u32 slots.
    // limbs[0] is the least significant 32 bits.
    limbs: [u32; FR_NUM_LIMBS],
}

#[repr(C, align(32))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
struct FrPacked64 {
    // Matches WGSL `struct Fr { limbs: vec4<u64> }` in `bn254_sum_u64.wgsl`.
    limbs: [u64; 4],
}

fn fr_to_limbs(x: Fr) -> FrLimbs {
    type FrBigInt = <Fr as PrimeField>::BigInt;
    // Use the internal Montgomery representation limbs (BigInt) directly, so
    // the GPU works over Montgomery residues just like arkworks does.
    let big: FrBigInt = x.0;
    let mut limbs = [0u32; FR_NUM_LIMBS];
    for (i, limb64) in big.0.iter().enumerate() {
        let lo = (*limb64 & 0xFFFF_FFFF) as u32;
        let hi = (*limb64 >> 32) as u32;
        limbs[2 * i] = lo;
        limbs[2 * i + 1] = hi;
    }
    FrLimbs { limbs }
}

fn limbs_to_fr(l: &FrLimbs) -> Fr {
    type FrBigInt = <Fr as PrimeField>::BigInt;
    let mut raw = [0u64; 4];
    for i in 0..4 {
        let lo = l.limbs[2 * i] as u64;
        let hi = (l.limbs[2 * i + 1] as u64) << 32;
        raw[i] = lo | hi;
    }
    let big = FrBigInt::new(raw);
    // Interpret the limbs as an internal Montgomery residue, matching the GPU.
    Fr::from_bigint_unchecked(big).expect("limbs_to_fr: value not in BN254 field (Montgomery)")
}

fn fr_to_packed64(x: Fr) -> FrPacked64 {
    type FrBigInt = <Fr as PrimeField>::BigInt;
    let big: FrBigInt = x.0;
    FrPacked64 { limbs: big.0 }
}

fn packed64_to_fr(p: &FrPacked64) -> Fr {
    type FrBigInt = <Fr as PrimeField>::BigInt;
    let big = FrBigInt::new(p.limbs);
    Fr::from_bigint_unchecked(big).expect("FrPacked64 -> Fr: value not in BN254 field")
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    len: u32,
    phase: u32,
    iters: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable)]
struct SumcheckParams {
    len: u32,
    _pad0: u32,
    r: FrPacked,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable)]
struct SumcheckParams64 {
    len: u32,
    _pad0: u32,
    r: FrPacked64,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
struct FrPacked {
    // Matches WGSL `struct Fr { limbs0: vec4<u32>, limbs1: vec4<u32> }`
    limbs0: [u32; 4],
    limbs1: [u32; 4],
}

impl From<FrLimbs> for FrPacked {
    fn from(l: FrLimbs) -> Self {
        FrPacked {
            limbs0: [l.limbs[0], l.limbs[1], l.limbs[2], l.limbs[3]],
            limbs1: [l.limbs[4], l.limbs[5], l.limbs[6], l.limbs[7]],
        }
    }
}

impl From<FrPacked> for FrLimbs {
    fn from(p: FrPacked) -> Self {
        FrLimbs {
            limbs: [
                p.limbs0[0],
                p.limbs0[1],
                p.limbs0[2],
                p.limbs0[3],
                p.limbs1[0],
                p.limbs1[1],
                p.limbs1[2],
                p.limbs1[3],
            ],
        }
    }
}

const SHADER_SRC_BN254: &str = include_str!("../../shaders/bn254_sum.wgsl");
const SHADER_SRC_BN254_U64: &str = include_str!("../../shaders/bn254_sum_u64.wgsl");

struct GpuBn254Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    params_buffer: wgpu::Buffer,
    // Sumcheck-specific
    sumcheck_bind_group_layout: wgpu::BindGroupLayout,
    sumcheck_eval_bind_pipeline: wgpu::ComputePipeline,
    sumcheck_params_buffer: wgpu::Buffer,
}

struct GpuBn254ContextU64 {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    params_buffer: wgpu::Buffer,
    // Sumcheck-specific
    sumcheck_bind_group_layout: wgpu::BindGroupLayout,
    sumcheck_eval_bind_pipeline: wgpu::ComputePipeline,
    sumcheck_params_buffer: wgpu::Buffer,
}

struct GpuBn254SumcheckState {
    p_in: wgpu::Buffer,
    p_out: wgpu::Buffer,
    q_in: wgpu::Buffer,
    q_out: wgpu::Buffer,
    g0_partial: wgpu::Buffer,
    g1_partial: wgpu::Buffer,
    g2_partial: wgpu::Buffer,
    current_len: u32,
}

fn as_bytes<T>(v: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((v as *const T) as *const u8, std::mem::size_of::<T>()) }
}

impl GpuBn254Context {
    async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapter found for BN254 experiment");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("bn254-sumcheck-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to create BN254 device");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bn254-sum-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC_BN254.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bn254-sum-bind-group-layout"),
            entries: &[
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
            label: Some("bn254-sum-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bn254-sum-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sum-params-buffer"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Sumcheck bind group layout mirrors the BN254 sumcheck bindings in WGSL.
        let sumcheck_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bn254-sumcheck-bind-group-layout"),
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
                label: Some("bn254-sumcheck-pipeline-layout"),
                bind_group_layouts: &[&sumcheck_bind_group_layout],
                push_constant_ranges: &[],
            });

        let sumcheck_eval_bind_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("bn254-sumcheck-eval-bind-pipeline"),
                layout: Some(&sumcheck_pipeline_layout),
                module: &shader,
                entry_point: "sumcheck_eval_bind_round",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let sumcheck_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sumcheck-params-buffer"),
            size: std::mem::size_of::<SumcheckParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            bind_group_layout,
            pipeline,
            params_buffer,
            sumcheck_bind_group_layout,
            sumcheck_eval_bind_pipeline,
            sumcheck_params_buffer,
        }
    }

    fn sumcheck_start(&self, p: &[Fr], q: &[Fr]) -> GpuBn254SumcheckState {
        assert!(!p.is_empty(), "sumcheck_start requires non-empty input");
        assert_eq!(
            p.len(),
            q.len(),
            "sumcheck_start: p and q must have same length"
        );
        let len = p.len();
        assert_eq!(
            len & (len - 1),
            0,
            "sumcheck_start: length must be power of two"
        );

        let encoded_p: Vec<FrPacked> = p
            .iter()
            .copied()
            .map(fr_to_limbs)
            .map(FrPacked::from)
            .collect();
        let encoded_q: Vec<FrPacked> = q
            .iter()
            .copied()
            .map(fr_to_limbs)
            .map(FrPacked::from)
            .collect();

        let element_size = std::mem::size_of::<FrPacked>() as u64;
        let buffer_size_bytes = (len as u64) * element_size;

        let p_in = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-p-in"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let p_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-p-out"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let q_in = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-q-in"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let q_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-q-out"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.queue
            .write_buffer(&p_in, 0, bytemuck::cast_slice(&encoded_p));
        self.queue
            .write_buffer(&q_in, 0, bytemuck::cast_slice(&encoded_q));

        let max_pairs = len / 2;
        let max_workgroups =
            ((max_pairs as u64) + (WORKGROUP_SIZE as u64) - 1) / (WORKGROUP_SIZE as u64);
        let partial_len = max_workgroups as usize;
        let partial_buffer_size = (partial_len as u64) * element_size;

        let g0_partial = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-g0-partial"),
            size: partial_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let g1_partial = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-g1-partial"),
            size: partial_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let g2_partial = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-g2-partial"),
            size: partial_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        GpuBn254SumcheckState {
            p_in,
            p_out,
            q_in,
            q_out,
            g0_partial,
            g1_partial,
            g2_partial,
            current_len: len as u32,
        }
    }

    /// Fused variant: for a given round j (len current_len, len/2 pairs), compute
    /// the degree-2 coefficients g0, g1, g2 *and* apply the verifier challenge r
    /// to bind p_j, q_j -> p_{j+1}, q_{j+1} on the GPU. The resulting
    /// p_{j+1}, q_{j+1} are stored in `state.p_in`, `state.q_in`, and
    /// `state.current_len` is halved.
    fn sumcheck_eval_bind_round(&self, state: &mut GpuBn254SumcheckState, r: Fr) -> (Fr, Fr, Fr) {
        assert!(
            state.current_len >= 2,
            "sumcheck_eval_bind_round requires at least one pair"
        );
        assert_eq!(
            state.current_len % 2,
            0,
            "current_len must be even in sumcheck_eval_bind_round"
        );

        let len = state.current_len;
        let pairs = len / 2;
        let workgroups = ((pairs as u64) + (WORKGROUP_SIZE as u64) - 1) / (WORKGROUP_SIZE as u64);
        let workgroups_u32 = workgroups as u32;

        let r_packed: FrPacked = fr_to_limbs(r).into();
        let params = SumcheckParams {
            len,
            _pad0: 0,
            r: r_packed,
        };
        self.queue
            .write_buffer(&self.sumcheck_params_buffer, 0, as_bytes(&params));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bn254-sumcheck-eval-bind-bind-group"),
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
                label: Some("bn254-sumcheck-eval-bind-encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bn254-sumcheck-eval-bind-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.sumcheck_eval_bind_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups_u32, 1, 1);
        }

        // Read back all per-workgroup partials and reduce on CPU.
        let element_size = std::mem::size_of::<FrPacked>() as u64;
        let partial_len = workgroups_u32 as u64;
        let partial_size_bytes = partial_len * element_size;

        let g0_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-g0-readback-eval-bind"),
            size: partial_size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let g1_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-g1-readback-eval-bind"),
            size: partial_size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let g2_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-g2-readback-eval-bind"),
            size: partial_size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&state.g0_partial, 0, &g0_readback, 0, partial_size_bytes);
        encoder.copy_buffer_to_buffer(&state.g1_partial, 0, &g1_readback, 0, partial_size_bytes);
        encoder.copy_buffer_to_buffer(&state.g2_partial, 0, &g2_readback, 0, partial_size_bytes);

        self.queue.submit(Some(encoder.finish()));

        let g0_slice = g0_readback.slice(..partial_size_bytes);
        let g1_slice = g1_readback.slice(..partial_size_bytes);
        let g2_slice = g2_readback.slice(..partial_size_bytes);

        let (g0_tx, g0_rx) = std::sync::mpsc::channel();
        g0_slice.map_async(wgpu::MapMode::Read, move |res| {
            g0_tx.send(res).ok();
        });
        let (g1_tx, g1_rx) = std::sync::mpsc::channel();
        g1_slice.map_async(wgpu::MapMode::Read, move |res| {
            g1_tx.send(res).ok();
        });
        let (g2_tx, g2_rx) = std::sync::mpsc::channel();
        g2_slice.map_async(wgpu::MapMode::Read, move |res| {
            g2_tx.send(res).ok();
        });

        self.device.poll(wgpu::Maintain::Wait);
        g0_rx
            .recv()
            .expect("map g0 fused")
            .expect("map g0 failed fused");
        g1_rx
            .recv()
            .expect("map g1 fused")
            .expect("map g1 failed fused");
        g2_rx
            .recv()
            .expect("map g2 fused")
            .expect("map g2 failed fused");

        let g0_data = g0_slice.get_mapped_range();
        let g1_data = g1_slice.get_mapped_range();
        let g2_data = g2_slice.get_mapped_range();

        let g0_partials: &[FrPacked] = bytemuck::cast_slice(&g0_data);
        let g1_partials: &[FrPacked] = bytemuck::cast_slice(&g1_data);
        let g2_partials: &[FrPacked] = bytemuck::cast_slice(&g2_data);

        let mut g0 = Fr::from(0u64);
        let mut g1 = Fr::from(0u64);
        let mut g2 = Fr::from(0u64);

        let partial_len_usize = partial_len as usize;
        for i in 0..partial_len_usize {
            g0 += limbs_to_fr(&FrLimbs::from(g0_partials[i]));
            g1 += limbs_to_fr(&FrLimbs::from(g1_partials[i]));
            g2 += limbs_to_fr(&FrLimbs::from(g2_partials[i]));
        }

        drop(g0_data);
        drop(g1_data);
        drop(g2_data);
        g0_readback.unmap();
        g1_readback.unmap();
        g2_readback.unmap();

        // Swap buffers and halve the current length, like sumcheck_apply_challenge.
        std::mem::swap(&mut state.p_in, &mut state.p_out);
        std::mem::swap(&mut state.q_in, &mut state.q_out);
        state.current_len = pairs;

        (g0, g1, g2)
    }
    fn sum_products_bn254(&self, p: &[Fr], q: &[Fr], iters: u32) -> Fr {
        assert_eq!(p.len(), q.len(), "p and q must have same length");
        let n = p.len();
        if n == 0 {
            return Fr::from(0u64);
        }

        let encoded_p: Vec<FrPacked> = p
            .iter()
            .copied()
            .map(fr_to_limbs)
            .map(FrPacked::from)
            .collect();
        let encoded_q: Vec<FrPacked> = q
            .iter()
            .copied()
            .map(fr_to_limbs)
            .map(FrPacked::from)
            .collect();

        let element_size = std::mem::size_of::<FrPacked>() as u64;
        let buffer_size_bytes = (n as u64) * element_size;

        let p_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-p-buffer"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let q_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-q-buffer"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-out-buffer"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.queue
            .write_buffer(&p_buf, 0, bytemuck::cast_slice(&encoded_p));
        self.queue
            .write_buffer(&q_buf, 0, bytemuck::cast_slice(&encoded_q));

        let mut current_len = n as u32;
        let mut input_buf = &p_buf;
        let mut output_buf = &out_buf;

        while current_len > 1 {
            let workgroups =
                ((current_len as u64) + (WORKGROUP_SIZE as u64) - 1) / (WORKGROUP_SIZE as u64);
            let workgroups_u32 = workgroups as u32;

            let params = Params {
                len: current_len,
                phase: if current_len == n as u32 { 0 } else { 1 },
                iters,
                _pad0: 0,
            };
            self.queue
                .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bn254-sum-bind-group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: q_buf.as_entire_binding(),
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
                    label: Some("bn254-sum-encoder"),
                });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("bn254-sum-pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(workgroups_u32, 1, 1);
            }

            self.queue.submit(Some(encoder.finish()));

            current_len = workgroups_u32;
            std::mem::swap(&mut input_buf, &mut output_buf);
        }

        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sum-result-buffer"),
            size: element_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bn254-sum-readback-encoder"),
            });
        encoder.copy_buffer_to_buffer(input_buf, 0, &result_buffer, 0, element_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = result_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("bn254: failed to receive map_async")
            .expect("bn254: failed to map result buffer");

        let data = slice.get_mapped_range();
        let packed: FrPacked = *bytemuck::from_bytes(&data[..std::mem::size_of::<FrPacked>()]);
        drop(data);
        result_buffer.unmap();

        limbs_to_fr(&FrLimbs::from(packed))
    }
}

impl GpuBn254ContextU64 {
    async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapter found for BN254 u64 experiment");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("bn254-sumcheck-device-u64"),
                    required_features: wgpu::Features::SHADER_INT64,
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to create BN254 u64 device");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bn254-sum-shader-u64"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC_BN254_U64.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bn254-sum-bind-group-layout-u64"),
            entries: &[
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
            label: Some("bn254-sum-pipeline-layout-u64"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bn254-sum-pipeline-u64"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sum-params-buffer-u64"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Sumcheck bind group layout mirrors the BN254 sumcheck bindings in WGSL.
        let sumcheck_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bn254-sumcheck-bind-group-layout-u64"),
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
                label: Some("bn254-sumcheck-pipeline-layout-u64"),
                bind_group_layouts: &[&sumcheck_bind_group_layout],
                push_constant_ranges: &[],
            });

        let sumcheck_eval_bind_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("bn254-sumcheck-eval-bind-pipeline-u64"),
                layout: Some(&sumcheck_pipeline_layout),
                module: &shader,
                entry_point: "sumcheck_eval_bind_round",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let sumcheck_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sumcheck-params-buffer-u64"),
            size: std::mem::size_of::<SumcheckParams64>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            bind_group_layout,
            pipeline,
            params_buffer,
            sumcheck_bind_group_layout,
            sumcheck_eval_bind_pipeline,
            sumcheck_params_buffer,
        }
    }

    fn sumcheck_start(&self, p: &[Fr], q: &[Fr]) -> GpuBn254SumcheckState {
        assert!(
            !p.is_empty(),
            "sumcheck_start (u64) requires non-empty input"
        );
        assert_eq!(
            p.len(),
            q.len(),
            "sumcheck_start (u64): p and q must have same length"
        );
        let len = p.len();
        assert_eq!(
            len & (len - 1),
            0,
            "sumcheck_start (u64): length must be power of two"
        );

        let encoded_p: Vec<FrPacked64> = p.iter().copied().map(fr_to_packed64).collect();
        let encoded_q: Vec<FrPacked64> = q.iter().copied().map(fr_to_packed64).collect();

        let element_size = std::mem::size_of::<FrPacked64>() as u64;
        let buffer_size_bytes = (len as u64) * element_size;

        let p_in = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-p-in-u64"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let p_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-p-out-u64"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let q_in = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-q-in-u64"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let q_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-q-out-u64"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.queue
            .write_buffer(&p_in, 0, bytemuck::cast_slice(&encoded_p));
        self.queue
            .write_buffer(&q_in, 0, bytemuck::cast_slice(&encoded_q));

        let max_pairs = len / 2;
        let max_workgroups =
            ((max_pairs as u64) + (WORKGROUP_SIZE as u64) - 1) / (WORKGROUP_SIZE as u64);
        let partial_len = max_workgroups as usize;
        let partial_buffer_size = (partial_len as u64) * element_size;

        let g0_partial = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-g0-partial-u64"),
            size: partial_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let g1_partial = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-g1-partial-u64"),
            size: partial_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let g2_partial = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-g2-partial-u64"),
            size: partial_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        GpuBn254SumcheckState {
            p_in,
            p_out,
            q_in,
            q_out,
            g0_partial,
            g1_partial,
            g2_partial,
            current_len: len as u32,
        }
    }

    /// Fused variant for the u64 path: compute g0, g1, g2 and apply the
    /// verifier challenge r to bind one variable in a single GPU pass.
    fn sumcheck_eval_bind_round(&self, state: &mut GpuBn254SumcheckState, r: Fr) -> (Fr, Fr, Fr) {
        assert!(
            state.current_len >= 2,
            "sumcheck_eval_bind_round (u64): need at least one pair"
        );
        assert_eq!(
            state.current_len % 2,
            0,
            "sumcheck_eval_bind_round (u64): len must be even"
        );

        let len = state.current_len;
        let pairs = len / 2;
        let workgroups = ((pairs as u64) + (WORKGROUP_SIZE as u64) - 1) / (WORKGROUP_SIZE as u64);
        let workgroups_u32 = workgroups as u32;

        let r_packed: FrPacked64 = fr_to_packed64(r);
        let params = SumcheckParams64 {
            len,
            _pad0: 0,
            r: r_packed,
        };
        self.queue
            .write_buffer(&self.sumcheck_params_buffer, 0, as_bytes(&params));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bn254-sumcheck-eval-bind-bind-group-u64"),
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
                label: Some("bn254-sumcheck-eval-bind-encoder-u64"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bn254-sumcheck-eval-bind-pass-u64"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.sumcheck_eval_bind_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups_u32, 1, 1);
        }

        // Read back all per-workgroup partials and reduce on CPU.
        let element_size = std::mem::size_of::<FrPacked64>() as u64;
        let partial_len = workgroups_u32 as u64;
        let partial_size_bytes = partial_len * element_size;

        let g0_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-g0-readback-eval-bind-u64"),
            size: partial_size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let g1_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-g1-readback-eval-bind-u64"),
            size: partial_size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let g2_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sc-g2-readback-eval-bind-u64"),
            size: partial_size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&state.g0_partial, 0, &g0_readback, 0, partial_size_bytes);
        encoder.copy_buffer_to_buffer(&state.g1_partial, 0, &g1_readback, 0, partial_size_bytes);
        encoder.copy_buffer_to_buffer(&state.g2_partial, 0, &g2_readback, 0, partial_size_bytes);

        self.queue.submit(Some(encoder.finish()));

        let g0_slice = g0_readback.slice(..partial_size_bytes);
        let g1_slice = g1_readback.slice(..partial_size_bytes);
        let g2_slice = g2_readback.slice(..partial_size_bytes);

        let (g0_tx, g0_rx) = std::sync::mpsc::channel();
        g0_slice.map_async(wgpu::MapMode::Read, move |res| {
            g0_tx.send(res).ok();
        });
        let (g1_tx, g1_rx) = std::sync::mpsc::channel();
        g1_slice.map_async(wgpu::MapMode::Read, move |res| {
            g1_tx.send(res).ok();
        });
        let (g2_tx, g2_rx) = std::sync::mpsc::channel();
        g2_slice.map_async(wgpu::MapMode::Read, move |res| {
            g2_tx.send(res).ok();
        });

        self.device.poll(wgpu::Maintain::Wait);
        g0_rx
            .recv()
            .expect("map g0 u64 fused")
            .expect("map g0 failed u64 fused");
        g1_rx
            .recv()
            .expect("map g1 u64 fused")
            .expect("map g1 failed u64 fused");
        g2_rx
            .recv()
            .expect("map g2 u64 fused")
            .expect("map g2 failed u64 fused");

        let g0_data = g0_slice.get_mapped_range();
        let g1_data = g1_slice.get_mapped_range();
        let g2_data = g2_slice.get_mapped_range();

        let g0_partials: &[FrPacked64] = bytemuck::cast_slice(&g0_data);
        let g1_partials: &[FrPacked64] = bytemuck::cast_slice(&g1_data);
        let g2_partials: &[FrPacked64] = bytemuck::cast_slice(&g2_data);

        let mut g0 = Fr::from(0u64);
        let mut g1 = Fr::from(0u64);
        let mut g2 = Fr::from(0u64);

        let partial_len_usize = partial_len as usize;
        for i in 0..partial_len_usize {
            g0 += packed64_to_fr(&g0_partials[i]);
            g1 += packed64_to_fr(&g1_partials[i]);
            g2 += packed64_to_fr(&g2_partials[i]);
        }

        drop(g0_data);
        drop(g1_data);
        drop(g2_data);
        g0_readback.unmap();
        g1_readback.unmap();
        g2_readback.unmap();

        std::mem::swap(&mut state.p_in, &mut state.p_out);
        std::mem::swap(&mut state.q_in, &mut state.q_out);
        state.current_len = pairs;

        (g0, g1, g2)
    }

    fn sum_products_bn254(&self, p: &[Fr], q: &[Fr], iters: u32) -> Fr {
        assert_eq!(p.len(), q.len(), "p and q must have same length (u64)");
        let n = p.len();
        if n == 0 {
            return Fr::from(0u64);
        }

        let encoded_p: Vec<FrPacked64> = p.iter().copied().map(fr_to_packed64).collect();
        let encoded_q: Vec<FrPacked64> = q.iter().copied().map(fr_to_packed64).collect();

        let element_size = std::mem::size_of::<FrPacked64>() as u64;
        let buffer_size_bytes = (n as u64) * element_size;

        let p_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-p-buffer-u64"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let q_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-q-buffer-u64"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-out-buffer-u64"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.queue
            .write_buffer(&p_buf, 0, bytemuck::cast_slice(&encoded_p));
        self.queue
            .write_buffer(&q_buf, 0, bytemuck::cast_slice(&encoded_q));

        let mut current_len = n as u32;
        let mut input_buf = &p_buf;
        let mut output_buf = &out_buf;

        while current_len > 1 {
            let workgroups =
                ((current_len as u64) + (WORKGROUP_SIZE as u64) - 1) / (WORKGROUP_SIZE as u64);
            let workgroups_u32 = workgroups as u32;

            let params = Params {
                len: current_len,
                phase: if current_len == n as u32 { 0 } else { 1 },
                iters,
                _pad0: 0,
            };
            self.queue
                .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bn254-sum-bind-group-u64"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: q_buf.as_entire_binding(),
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
                    label: Some("bn254-sum-encoder-u64"),
                });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("bn254-sum-pass-u64"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(workgroups_u32, 1, 1);
            }

            self.queue.submit(Some(encoder.finish()));

            current_len = workgroups_u32;
            std::mem::swap(&mut input_buf, &mut output_buf);
        }

        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-sum-result-buffer-u64"),
            size: element_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bn254-sum-readback-encoder-u64"),
            });
        encoder.copy_buffer_to_buffer(input_buf, 0, &result_buffer, 0, element_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = result_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("bn254 u64: failed to receive map_async")
            .expect("bn254 u64: failed to map result buffer");

        let data = slice.get_mapped_range();
        let packed: FrPacked64 = *bytemuck::from_bytes(&data[..std::mem::size_of::<FrPacked64>()]);
        drop(data);
        result_buffer.unmap();

        packed64_to_fr(&packed)
    }
}

fn main() {
    // Minimal BN254 inner-product experiment:
    //  - Sample random Fr vectors p, q
    //  - Compute sum_i p_i * q_i on CPU
    //  - Compute the same sum on GPU using bn254_sum.wgsl
    //  - Compare for equality and print timings.
    let mut args = std::env::args();
    args.next(); // program name
    let log2_len: u32 = args
        .next()
        .and_then(|s| {
            let cleaned: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
            if cleaned.is_empty() {
                None
            } else {
                cleaned.parse().ok()
            }
        })
        .unwrap_or(20);
    assert!(log2_len <= 24, "log2_len too large for bn254 demo");
    let n: usize = 1usize << log2_len;

    // Optional second CLI arg: number of repeated multiplications per pair.
    // Example:
    //   cargo run -p webgpu-sumcheck --bin bn254_sumcheck -- 20 1024
    let iters: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    let mut rng = rand::thread_rng();
    let p: Vec<Fr> = (0..n).map(|_| Fr::from(rng.gen::<u64>())).collect();
    let q: Vec<Fr> = (0..n).map(|_| Fr::from(rng.gen::<u64>())).collect();

    // CPU baselines: sequential and Rayon-parallel.
    // We perform `iters` multiplications per input pair, matching the GPU work.
    let start_cpu_seq = Instant::now();
    let cpu_sum_seq: Fr = p
        .iter()
        .zip(q.iter())
        .map(|(a, b)| {
            let prod = *a * *b;
            let mut acc = Fr::from(0u64);
            for _ in 0..iters {
                acc += prod;
            }
            acc
        })
        .fold(Fr::from(0u64), |acc, x| acc + x);
    let cpu_time_seq = start_cpu_seq.elapsed();

    let start_cpu_par = Instant::now();
    let cpu_sum_par: Fr = p
        .par_iter()
        .zip(&q)
        .map(|(a, b)| {
            let prod = *a * *b;
            let mut acc = Fr::from(0u64);
            for _ in 0..iters {
                acc += prod;
            }
            acc
        })
        .reduce(|| Fr::from(0u64), |acc, x| acc + x);
    let cpu_time_par = start_cpu_par.elapsed();

    let start_gpu_setup = Instant::now();
    let ctx = pollster::block_on(GpuBn254Context::new());
    let gpu_setup_time = start_gpu_setup.elapsed();

    let start_gpu_u32 = Instant::now();
    let gpu_sum_u32 = ctx.sum_products_bn254(&p, &q, iters);
    let gpu_time_u32 = start_gpu_u32.elapsed();

    let start_gpu_setup_u64 = Instant::now();
    let ctx_u64 = pollster::block_on(GpuBn254ContextU64::new());
    let gpu_setup_time_u64 = start_gpu_setup_u64.elapsed();

    let start_gpu_u64 = Instant::now();
    let gpu_sum_u64 = ctx_u64.sum_products_bn254(&p, &q, iters);
    let gpu_time_u64 = start_gpu_u64.elapsed();

    println!("BN254 inner-product experiment");
    println!("log2_len: {}", log2_len);
    println!("length: {}", n);
    println!("iters per pair: {}", iters);
    println!("CPU sum (seq): {:?}", cpu_sum_seq);
    println!("CPU sum (par): {:?}", cpu_sum_par);
    println!("GPU sum (8×u32): {:?}", gpu_sum_u32);
    println!("GPU sum (4×u64): {:?}", gpu_sum_u64);
    let match_all =
        cpu_sum_seq == cpu_sum_par && cpu_sum_seq == gpu_sum_u32 && cpu_sum_seq == gpu_sum_u64;
    println!(
        "Match (CPU seq vs CPU par vs 8×u32 vs 4×u64): {}",
        match_all
    );
    println!("CPU 1-thread time: {:?}", cpu_time_seq);
    println!("CPU Rayon time:    {:?}", cpu_time_par);
    println!("GPU setup time (8×u32): {:?}", gpu_setup_time);
    println!("GPU sum time (8×u32): {:?}", gpu_time_u32);
    println!("GPU setup time (4×u64): {:?}", gpu_setup_time_u64);
    println!("GPU sum time (4×u64): {:?}", gpu_time_u64);
    let total_mul = (n as f64) * (iters as f64);
    let cpu_seq_mps = total_mul / cpu_time_seq.as_secs_f64() / 1e6;
    let cpu_par_mps = total_mul / cpu_time_par.as_secs_f64() / 1e6;
    let gpu_u32_mps = total_mul / gpu_time_u32.as_secs_f64() / 1e6;
    let gpu_u64_mps = total_mul / gpu_time_u64.as_secs_f64() / 1e6;
    println!("CPU 1-thread mult throughput: {:.3} Mmul/s", cpu_seq_mps);
    println!("CPU Rayon mult throughput:    {:.3} Mmul/s", cpu_par_mps);
    println!(
        "GPU mult throughput (8×u32 limbs): {:.3} Mmul/s",
        gpu_u32_mps
    );
    println!(
        "GPU mult throughput (4×u64 limbs): {:.3} Mmul/s",
        gpu_u64_mps
    );

    // BN254 sumcheck experiment (minimal, re-uses the same p, q).
    if log2_len <= 16 {
        let mut sc_state = ctx.sumcheck_start(&p, &q);
        let mut sc_state_u64 = ctx_u64.sumcheck_start(&p, &q);

        let mut p_sc = p.clone();
        let mut q_sc = q.clone();
        let mut len_sc = n;
        let mut rng_ch = rand::thread_rng();

        let mut ok = true;

        for round in 0..log2_len {
            // CPU reference for this round, using the current p_sc, q_sc (p_j, q_j).
            let slice_len = len_sc;
            let pairs = slice_len / 2;
            let mut g0_cpu = Fr::from(0u64);
            let mut g1_cpu = Fr::from(0u64);
            let mut g2_cpu = Fr::from(0u64);
            for i in 0..pairs {
                let p0 = p_sc[2 * i];
                let p1 = p_sc[2 * i + 1];
                let q0 = q_sc[2 * i];
                let q1 = q_sc[2 * i + 1];
                let dp = p1 - p0;
                let dq = q1 - q0;
                let a = p0 * q0;
                let b = p0 * dq + q0 * dp;
                let c = dp * dq;
                g0_cpu += a;
                g1_cpu += b;
                g2_cpu += c;
            }

            // Sample verifier challenge r_j on CPU.
            let r_j: Fr = Fr::from(rng_ch.gen::<u64>());

            // GPU: fused eval + bind for this round using the same r_j.
            let (g0_gpu, g1_gpu, g2_gpu) = ctx.sumcheck_eval_bind_round(&mut sc_state, r_j);
            let (g0_gpu_u64, g1_gpu_u64, g2_gpu_u64) =
                ctx_u64.sumcheck_eval_bind_round(&mut sc_state_u64, r_j);

            if g0_cpu != g0_gpu
                || g1_cpu != g1_gpu
                || g2_cpu != g2_gpu
                || g0_cpu != g0_gpu_u64
                || g1_cpu != g1_gpu_u64
                || g2_cpu != g2_gpu_u64
            {
                println!(
                    "sumcheck round {} mismatch:\n  CPU   ({:?}, {:?}, {:?})\n  GPU32({:?}, {:?}, {:?})\n  GPU64({:?}, {:?}, {:?})",
                    round,
                    g0_cpu,
                    g1_cpu,
                    g2_cpu,
                    g0_gpu,
                    g1_gpu,
                    g2_gpu,
                    g0_gpu_u64,
                    g1_gpu_u64,
                    g2_gpu_u64
                );
                ok = false;
                break;
            }

            // CPU bind
            for i in 0..pairs {
                let p0 = p_sc[2 * i];
                let p1 = p_sc[2 * i + 1];
                let q0 = q_sc[2 * i];
                let q1 = q_sc[2 * i + 1];
                let dp = p1 - p0;
                let dq = q1 - q0;
                p_sc[i] = p0 + r_j * dp;
                q_sc[i] = q0 + r_j * dq;
            }
            len_sc /= 2;
        }

        println!("BN254 sumcheck round-by-round match: {}", ok);
    }
}
