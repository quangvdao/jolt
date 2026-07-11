use std::time::Instant;

use ark_bn254::Fr;
use ark_ff::PrimeField;
use rand::Rng;
use rayon::prelude::*;

const FR_NUM_LIMBS: usize = 8; // 8 × 32-bit limbs = 256 bits for BN254 Fr
const WORKGROUP_SIZE: u32 = 256;
const NUM_SAMPLING: usize = 192;

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

#[repr(C, align(32))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
struct GlobalConfigPacked {
    // Matches WGSL `global_config_t` layout in `bignum.wgsl`:
    //   p, double_p, J, barrett_factor, constant : bigint (each 32 bytes)
    p: FrPacked,
    double_p: FrPacked,
    J: FrPacked,
    barrett_factor: FrPacked,
    constant: FrPacked,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
struct InnerParams {
    // Matches WGSL `inner_params_t` used by `bn254_inner_product` in `bignum.wgsl`.
    len: u32,
    phase: u32,
    iters: u32,
    _pad0: u32,
}

fn frpacked_from_u32_le(limbs: [u32; FR_NUM_LIMBS]) -> FrPacked {
    FrPacked {
        limbs0: [limbs[0], limbs[1], limbs[2], limbs[3]],
        limbs1: [limbs[4], limbs[5], limbs[6], limbs[7]],
    }
}

fn bn254_global_config() -> GlobalConfigPacked {
    // Constants derived once via a small Python script (see tooling notes).
    // All values are little-endian 32-bit limbs.
    const P_U32_LE: [u32; FR_NUM_LIMBS] = [
        0xf0000001,
        0x43e1f593,
        0x79b97091,
        0x2833e848,
        0x8181585d,
        0xb85045b6,
        0xe131a029,
        0x30644e72,
    ];
    const DOUBLE_P_U32_LE: [u32; FR_NUM_LIMBS] = [
        0xe0000002,
        0x87c3eb27,
        0xf372e122,
        0x5067d090,
        0x0302b0ba,
        0x70a08b6d,
        0xc2634053,
        0x60c89ce5,
    ];
    const J_U32_LE: [u32; FR_NUM_LIMBS] = [
        0xefffffff,
        0xc2e1f593,
        0x4c6911b3,
        0x6586864b,
        0x99062391,
        0xe39a9828,
        0x0d8341b2,
        0x73f82f1d,
    ];

    GlobalConfigPacked {
        p: frpacked_from_u32_le(P_U32_LE),
        double_p: frpacked_from_u32_le(DOUBLE_P_U32_LE),
        J: frpacked_from_u32_le(J_U32_LE),
        // For this experiment we do not use Barrett-based kernels, so set
        // barrett_factor and constant to zero.
        barrett_factor: FrPacked {
            limbs0: [0u32; 4],
            limbs1: [0u32; 4],
        },
        constant: FrPacked {
            limbs0: [0u32; 4],
            limbs1: [0u32; 4],
        },
    }
}

const SHADER_SRC_BN254: &str = include_str!("../shaders/bn254_sum.wgsl");
const SHADER_SRC_BN254_U64: &str = include_str!("../shaders/bn254_sum_u64.wgsl");
const SHADER_SRC_BIGNUM: &str = include_str!("../shaders/bignum.wgsl");

struct GpuBn254Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    params_buffer: wgpu::Buffer,
}

struct GpuBn254ContextU64 {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    params_buffer: wgpu::Buffer,
}

struct GpuBn254ContextBigint {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout0: wgpu::BindGroupLayout,
    bind_group_layout1: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    global_config_buffer: wgpu::Buffer,
    sample_index_buffer: wgpu::Buffer,
    inner_params_buffer: wgpu::Buffer,
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
                    label: Some("bn254-sum-device"),
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

        Self {
            device,
            queue,
            bind_group_layout,
            pipeline,
            params_buffer,
        }
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
                    label: Some("bn254-sum-device-u64"),
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

        Self {
            device,
            queue,
            bind_group_layout,
            pipeline,
            params_buffer,
        }
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

impl GpuBn254ContextBigint {
    async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapter found for BN254 bigint experiment");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("bn254-bignum-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to create BN254 bignum device");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bn254-bignum-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC_BIGNUM.into()),
        });

        // group(0): global_config, sample_index, inner_params
        let bind_group_layout0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bn254-bignum-bind-group-layout0"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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

        // group(1): vector_x (@binding 1), vector_y (@binding 2), vector_out (@binding 3)
        let bind_group_layout1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bn254-bignum-bind-group-layout1"),
                entries: &[
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bn254-bignum-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout0, &bind_group_layout1],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bn254-bignum-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "bn254_inner_product",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let global_config = bn254_global_config();
        let global_config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-bignum-global-config"),
            size: std::mem::size_of::<GlobalConfigPacked>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &global_config_buffer,
            0,
            bytemuck::bytes_of(&global_config),
        );

        // sample_index is unused for the inner-product kernel, but the binding
        // exists in the shader, so bind a zeroed buffer.
        let sample_index: Vec<[u32; 4]> = vec![[0u32; 4]; NUM_SAMPLING];
        let sample_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-bignum-sample-index"),
            size: (sample_index.len() * std::mem::size_of::<[u32; 4]>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &sample_index_buffer,
            0,
            bytemuck::cast_slice(&sample_index),
        );

        let inner_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-bignum-inner-params"),
            size: std::mem::size_of::<InnerParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            bind_group_layout0,
            bind_group_layout1,
            pipeline,
            global_config_buffer,
            sample_index_buffer,
            inner_params_buffer,
        }
    }

    fn sum_products_bn254(&self, p: &[Fr], q: &[Fr], iters: u32) -> Fr {
        assert_eq!(p.len(), q.len(), "p and q must have same length (bignum)");
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

        let vec_x = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-bignum-x"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let vec_y = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-bignum-y"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let vec_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-bignum-out"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.queue
            .write_buffer(&vec_x, 0, bytemuck::cast_slice(&encoded_p));
        self.queue
            .write_buffer(&vec_y, 0, bytemuck::cast_slice(&encoded_q));

        let mut current_len = n as u32;
        let mut input_buf = &vec_x;
        let mut output_buf = &vec_out;

        // First pass: do multiplies and per-workgroup reduction.
        if current_len > 1 {
            let workgroups = ((current_len as u64) + (WORKGROUP_SIZE as u64) - 1)
                / (WORKGROUP_SIZE as u64);
            let workgroups_u32 = workgroups as u32;

            let params = InnerParams {
                len: current_len,
                phase: 0,
                iters,
                _pad0: 0,
            };
            self.queue.write_buffer(
                &self.inner_params_buffer,
                0,
                bytemuck::bytes_of(&params),
            );

            let bind_group0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bn254-bignum-bind-group0"),
                layout: &self.bind_group_layout0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.global_config_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.sample_index_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.inner_params_buffer.as_entire_binding(),
                    },
                ],
            });

            let bind_group1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bn254-bignum-bind-group1-initial"),
                layout: &self.bind_group_layout1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vec_x.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: vec_y.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: vec_out.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bn254-bignum-encoder-initial"),
                });

            {
                let mut cpass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("bn254-bignum-pass-initial"),
                        timestamp_writes: None,
                    });
                cpass.set_pipeline(&self.pipeline);
                cpass.set_bind_group(0, &bind_group0, &[]);
                cpass.set_bind_group(1, &bind_group1, &[]);
                cpass.dispatch_workgroups(workgroups_u32, 1, 1);
            }

            self.queue.submit(Some(encoder.finish()));

            current_len = workgroups_u32;
            input_buf = &vec_out;
            output_buf = &vec_x;
        }

        // Subsequent passes: just reductions over the accumulated partial sums.
        while current_len > 1 {
            let workgroups = ((current_len as u64) + (WORKGROUP_SIZE as u64) - 1)
                / (WORKGROUP_SIZE as u64);
            let workgroups_u32 = workgroups as u32;

            let params = InnerParams {
                len: current_len,
                phase: 1,
                iters: 0,
                _pad0: 0,
            };
            self.queue.write_buffer(
                &self.inner_params_buffer,
                0,
                bytemuck::bytes_of(&params),
            );

            let bind_group0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bn254-bignum-bind-group0-reduce"),
                layout: &self.bind_group_layout0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.global_config_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.sample_index_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.inner_params_buffer.as_entire_binding(),
                    },
                ],
            });

            let bind_group1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bn254-bignum-bind-group1-reduce"),
                layout: &self.bind_group_layout1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input_buf.as_entire_binding(),
                    },
                    // y is unused in phase != 0, but must be bound.
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: vec_y.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bn254-bignum-encoder-reduce"),
                });

            {
                let mut cpass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("bn254-bignum-pass-reduce"),
                        timestamp_writes: None,
                    });
                cpass.set_pipeline(&self.pipeline);
                cpass.set_bind_group(0, &bind_group0, &[]);
                cpass.set_bind_group(1, &bind_group1, &[]);
                cpass.dispatch_workgroups(workgroups_u32, 1, 1);
            }

            self.queue.submit(Some(encoder.finish()));

            current_len = workgroups_u32;
            std::mem::swap(&mut input_buf, &mut output_buf);
        }

        // Read back the single remaining Fr element from input_buf[0].
        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bn254-bignum-result-buffer"),
            size: element_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bn254-bignum-readback-encoder"),
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
            .expect("bn254 bignum: failed to receive map_async")
            .expect("bn254 bignum: failed to map result buffer");

        let data = slice.get_mapped_range();
        let packed: FrPacked = *bytemuck::from_bytes(&data[..std::mem::size_of::<FrPacked>()]);
        drop(data);
        result_buffer.unmap();

        let limbs = FrLimbs::from(packed);
        limbs_to_fr(&limbs)
    }
}

fn main() {
    // BN254 inner-product experiment:
    //  - Sample random Fr vectors p, q
    //  - Compute sum_i p_i * q_i on CPU
    //  - Compute the same sum on GPU using bn254_sum.wgsl (32-bit and 64-bit limbs)
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
    //   cargo run -p webgpu-sumcheck --example gpu_vs_cpu_bn254 -- 20 1024
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
    let ctx_bignum = pollster::block_on(GpuBn254ContextBigint::new());
    let gpu_setup_time = start_gpu_setup.elapsed();

    let start_gpu_bignum = Instant::now();
    let gpu_sum_bignum = ctx_bignum.sum_products_bn254(&p, &q, iters);
    let gpu_time_bignum = start_gpu_bignum.elapsed();

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
    println!("GPU sum (bignum bigint): {:?}", gpu_sum_bignum);
    println!("GPU sum (4×u64): {:?}", gpu_sum_u64);
    let match_all = cpu_sum_seq == cpu_sum_par
        && cpu_sum_seq == gpu_sum_bignum
        && cpu_sum_seq == gpu_sum_u64;
    println!(
        "Match (CPU seq vs CPU par vs 8×u32 vs 4×u64): {}",
        match_all
    );
    println!("CPU 1-thread time: {:?}", cpu_time_seq);
    println!("CPU Rayon time:    {:?}", cpu_time_par);
    println!("GPU setup time (bignum): {:?}", gpu_setup_time);
    println!("GPU sum time (bignum): {:?}", gpu_time_bignum);
    println!("GPU setup time (4×u64): {:?}", gpu_setup_time_u64);
    println!("GPU sum time (4×u64): {:?}", gpu_time_u64);
    let total_mul = (n as f64) * (iters as f64);
    let cpu_seq_mps = total_mul / cpu_time_seq.as_secs_f64() / 1e6;
    let cpu_par_mps = total_mul / cpu_time_par.as_secs_f64() / 1e6;
    let gpu_bignum_mps = total_mul / gpu_time_bignum.as_secs_f64() / 1e6;
    let gpu_u64_mps = total_mul / gpu_time_u64.as_secs_f64() / 1e6;
    println!("CPU 1-thread mult throughput: {:.3} Mmul/s", cpu_seq_mps);
    println!("CPU Rayon mult throughput:    {:.3} Mmul/s", cpu_par_mps);
    println!(
        "GPU mult throughput (bignum bigint): {:.3} Mmul/s",
        gpu_bignum_mps
    );
    println!(
        "GPU mult throughput (4×u64 limbs): {:.3} Mmul/s",
        gpu_u64_mps
    );
}


