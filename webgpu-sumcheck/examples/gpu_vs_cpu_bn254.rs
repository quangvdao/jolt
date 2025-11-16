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

const SHADER_SRC_BN254: &str = include_str!("../shaders/bn254_sum.wgsl");
const SHADER_SRC_BN254_U64: &str = include_str!("../shaders/bn254_sum_u64.wgsl");

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
}


