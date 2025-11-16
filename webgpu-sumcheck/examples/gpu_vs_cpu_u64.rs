use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;

use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;

// WGSL compute shader performing many logical u64 ops per element (half adds,
// half multiplies) using a pair of u32 limbs (little-endian: [lo, hi]).
const SHADER_SRC: &str = r#"
struct BufferU64 {
    data: array<vec2<u32>>,
};

struct Params {
    len: u32,
    iters: u32,
};

struct MulWideResult {
    lo: u32,
    hi: u32,
};

fn mul_wide(a: u32, b: u32) -> MulWideResult {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;

    let carry = p0 >> 16u;
    let mid = carry + (p1 & 0xFFFFu) + (p2 & 0xFFFFu);

    let lo = (p0 & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);
    let hi = (mid >> 16u) + (p1 >> 16u) + (p2 >> 16u) + p3;

    var res: MulWideResult;
    res.lo = lo;
    res.hi = hi;
    return res;
}

fn mul_u64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    // a = a0 + 2^32 a1, b = b0 + 2^32 b1
    // a*b mod 2^64 = p0 + 2^32 * cross, where
    //   p0 = a0*b0 (64-bit)
    //   cross = a0*b1 + a1*b0 (we only need this mod 2^32)
    let p0 = mul_wide(a.x, b.x);
    let cross = a.x * b.y + a.y * b.x; // automatically taken mod 2^32

    let lo = p0.lo;
    let hi = p0.hi + cross; // overflow ignored mod 2^32
    return vec2<u32>(lo, hi);
}

fn add_u64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry = select(0u, 1u, lo < a.x);
    let hi = a.y + b.y + carry;
    return vec2<u32>(lo, hi);
}

@group(0) @binding(0)
var<storage, read> in_a: BufferU64;

@group(0) @binding(1)
var<storage, read> in_b: BufferU64;

@group(0) @binding(2)
var<storage, read_write> out_buf: BufferU64;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }

    var acc: vec2<u32> = in_a.data[idx];
    let b: vec2<u32> = in_b.data[idx];
    let half: u32 = params.iters / 2u;

    // Do a lot of arithmetic per element so we are compute-bound rather than
    // dominated by copy overhead.
    var i: u32 = 0u;
    loop {
        if (i >= params.iters) {
            break;
        }
        if (i < half) {
            acc = add_u64(acc, b);
        } else {
            acc = mul_u64(acc, b);
        }
        i = i + 1u;
    }

    out_buf.data[idx] = acc;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    len: u32,
    iters: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct U64Words {
    limbs: [u32; 2],
}

fn u64_to_words(x: u64) -> U64Words {
    U64Words {
        limbs: [x as u32, (x >> 32) as u32],
    }
}

fn words_to_u64(w: &U64Words) -> u64 {
    (w.limbs[0] as u64) | ((w.limbs[1] as u64) << 32)
}

fn main() {
    // CLI:
    //   cargo run -p webgpu-sumcheck --example gpu_vs_cpu_u64 -- [len] [iters]
    //
    // Defaults are chosen to be heavy enough to show a clear GPU win on a
    // modern GPU, while remaining safe in memory usage.
    let mut args = std::env::args();
    args.next(); // program name

    let len: usize = args
        .next()
        .map(|s| s.parse().expect("len must be a positive integer"))
        .unwrap_or(1usize << 23); // 8,388,608 elements by default

    let iters: u32 = args
        .next()
        .map(|s| s.parse().expect("iters must be a positive integer"))
        .unwrap_or(1024);

    assert!(
        len > 0 && len <= (1usize << 26),
        "len out of range (1 ..= 2^26 recommended)"
    );

    println!("Benchmarking u64 add+mul (half adds, half muls):");
    println!("  len   = {}", len);
    println!("  iters = {}", iters);

    // Generate inputs once and reuse for both CPU and GPU.
    let mut rng = rand::thread_rng();
    let a: Vec<u64> = (0..len).map(|_| rng.gen()).collect();
    let b: Vec<u64> = (0..len).map(|_| rng.gen()).collect();

    let a_words: Vec<U64Words> = a.iter().copied().map(u64_to_words).collect();
    let b_words: Vec<U64Words> = b.iter().copied().map(u64_to_words).collect();

    // -------------------------
    // CPU (single-threaded)
    // -------------------------
    let start_cpu = Instant::now();
    let mut out_cpu = vec![0u64; len];
    let half = iters / 2;
    for i in 0..len {
        let mut acc = a[i];
        let bv = b[i];
        for j in 0..iters {
            if j < half {
                acc = acc.wrapping_add(bv);
            } else {
                acc = acc.wrapping_mul(bv);
            }
        }
        out_cpu[i] = acc;
    }
    let cpu_time = start_cpu.elapsed();

    // -------------------------
    // CPU (multi-threaded / Rayon)
    // -------------------------
    let start_cpu_mt = Instant::now();
    let half_mt = iters / 2;
    let out_cpu_mt: Vec<u64> = a
        .par_iter()
        .zip(&b)
        .map(|(&av, &bv)| {
            let mut acc = av;
            for j in 0..iters {
                if j < half_mt {
                    acc = acc.wrapping_add(bv);
                } else {
                    acc = acc.wrapping_mul(bv);
                }
            }
            acc
        })
        .collect();
    let cpu_mt_time = start_cpu_mt.elapsed();

    assert_eq!(
        out_cpu, out_cpu_mt,
        "single-threaded and Rayon results differ"
    );

    // -------------------------
    // GPU setup
    // -------------------------
    let start_gpu_setup = Instant::now();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("No suitable GPU adapter found");

    let info = adapter.get_info();
    println!(
        "Using adapter: {} (backend: {:?}, device_type: {:?})",
        info.name, info.backend, info.device_type
    );

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("gpu-vs-cpu-u64-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("Failed to create device");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("u64-add-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gpu-vs-cpu-u64-bgl"),
        entries: &[
            // in_a
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
            // in_b
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
            // out_buf
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
        label: Some("gpu-vs-cpu-u64-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gpu-vs-cpu-u64-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let buffer_size_bytes = (len as u64) * std::mem::size_of::<U64Words>() as u64;

    let a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("in_a-buffer"),
        contents: bytemuck::cast_slice(&a_words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("in_b-buffer"),
        contents: bytemuck::cast_slice(&b_words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out-buffer"),
        size: buffer_size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = Params {
        len: len as u32,
        iters,
        _pad0: 0,
        _pad1: 0,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params-buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-vs-cpu-u64-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback-buffer"),
        size: buffer_size_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let gpu_setup_time = start_gpu_setup.elapsed();

    // -------------------------
    // GPU compute + readback
    // -------------------------
    let workgroup_count = ((len as u32) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let max_workgroups = device.limits().max_compute_workgroups_per_dimension;
    assert!(
        workgroup_count <= max_workgroups,
        "len too large for this device with WORKGROUP_SIZE {} (workgroups={} > max={})",
        WORKGROUP_SIZE,
        workgroup_count,
        max_workgroups
    );

    let start_gpu = Instant::now();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-vs-cpu-u64-encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu-vs-cpu-u64-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&out_buffer, 0, &readback_buffer, 0, buffer_size_bytes);

    queue.submit(Some(encoder.finish()));

    let gpu_submit_time = start_gpu.elapsed();

    // Map and read back results.
    let buffer_slice = readback_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
        tx.send(res).ok();
    });

    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .expect("Failed to receive map_async result")
        .expect("Failed to map readback buffer");

    let data = buffer_slice.get_mapped_range();
    let words: &[U64Words] = bytemuck::cast_slice(&data);
    let out_gpu: Vec<u64> = words.iter().map(words_to_u64).collect();
    drop(data);
    readback_buffer.unmap();

    let gpu_total_time = start_gpu.elapsed();

    assert_eq!(out_cpu, out_gpu, "CPU and GPU results differ");

    let total_ops: f64 = (len as f64) * (iters as f64);
    let cpu_ops_per_sec = total_ops / cpu_time.as_secs_f64();
    let cpu_mt_ops_per_sec = total_ops / cpu_mt_time.as_secs_f64();
    let gpu_ops_per_sec = total_ops / gpu_total_time.as_secs_f64();

    println!();
    println!(
        "Results (all perform {} u64 ops: half add, half mul):",
        total_ops as u64
    );
    println!(
        "  CPU 1-thread:   {:?}  ({:.3} Gops/s)",
        cpu_time,
        cpu_ops_per_sec / 1e9
    );
    println!(
        "  CPU Rayon MT:   {:?}  ({:.3} Gops/s)",
        cpu_mt_time,
        cpu_mt_ops_per_sec / 1e9
    );
    println!(
        "  GPU kernel:     {:?}  ({:.3} Gops/s)",
        gpu_total_time,
        gpu_ops_per_sec / 1e9
    );
    println!("  GPU setup time: {:?}", gpu_setup_time);
    println!("  GPU submit-only (no map): {:?}", gpu_submit_time);
}
