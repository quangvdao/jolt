use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;

use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;

// A very simple WGSL compute shader that performs many u32 ops per element:
// half additions, half multiplications.
const SHADER_SRC: &str = r#"
struct BufferU32 {
    data: array<u32>,
};

struct Params {
    len: u32,
    iters: u32,
};

@group(0) @binding(0)
var<storage, read> in_a: BufferU32;

@group(0) @binding(1)
var<storage, read> in_b: BufferU32;

@group(0) @binding(2)
var<storage, read_write> out_buf: BufferU32;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }

    var acc: u32 = in_a.data[idx];
    let b: u32 = in_b.data[idx];
    let half: u32 = params.iters / 2u;

    // Do a lot of arithmetic per element so we are compute-bound rather than
    // dominated by PCIe / copy overhead. For the first half of the iterations
    // we add, for the second half we multiply.
    var i: u32 = 0u;
    loop {
        if (i >= params.iters) {
            break;
        }
        if (i < half) {
            acc = acc + b;
        } else {
            acc = acc * b;
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

fn main() {
    // CLI:
    //   cargo run -p webgpu-sumcheck --example gpu_vs_cpu_u32 -- [len] [iters]
    //
    // Defaults are chosen to be "heavy enough" that the GPU wins clearly on a
    // discrete GPU, while still fitting easily in memory.
    let mut args = std::env::args();
    args.next(); // program name

    let len: usize = args
        .next()
        .map(|s| s.parse().expect("len must be a positive integer"))
        .unwrap_or(1usize << 23); // 8,388,608 elements

    let iters: u32 = args
        .next()
        .map(|s| s.parse().expect("iters must be a positive integer"))
        .unwrap_or(1024);

    // Keep sizes in a safe range.
    assert!(
        len > 0 && len <= (1usize << 26),
        "len out of range (1 ..= 2^26 recommended)"
    );

    println!("Benchmarking u32 add+mul (half adds, half muls):");
    println!("  len   = {}", len);
    println!("  iters = {}", iters);

    // Generate inputs once and reuse for both CPU and GPU.
    let mut rng = rand::thread_rng();
    let a: Vec<u32> = (0..len).map(|_| rng.gen()).collect();
    let b: Vec<u32> = (0..len).map(|_| rng.gen()).collect();

    // -------------------------
    // CPU (single-threaded)
    // -------------------------
    let start_cpu = Instant::now();
    let mut out_cpu = vec![0u32; len];
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
    let out_cpu_mt: Vec<u32> = a
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

    assert_eq!(out_cpu, out_cpu_mt, "single-threaded and Rayon results differ");

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
            label: Some("gpu-vs-cpu-u32-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("Failed to create device");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("u32-add-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gpu-vs-cpu-u32-bgl"),
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
        label: Some("gpu-vs-cpu-u32-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gpu-vs-cpu-u32-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let buffer_size_bytes = (len as u64) * std::mem::size_of::<u32>() as u64;

    let a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("in_a-buffer"),
        contents: bytemuck::cast_slice(&a),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("in_b-buffer"),
        contents: bytemuck::cast_slice(&b),
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
        label: Some("gpu-vs-cpu-u32-bind-group"),
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
    let workgroup_count =
        ((len as u32) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
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
        label: Some("gpu-vs-cpu-u32-encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu-vs-cpu-u32-pass"),
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
    let out_gpu: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
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
        "Results (all perform {} u32 ops: half add, half mul):",
        total_ops as u64
    );
    println!("  CPU 1-thread:   {:?}  ({:.3} Gops/s)", cpu_time, cpu_ops_per_sec / 1e9);
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


