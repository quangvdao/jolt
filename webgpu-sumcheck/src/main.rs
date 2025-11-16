fn main() {
    eprintln!("webgpu-sumcheck: This package provides multiple binaries and examples.");
    eprintln!();
    eprintln!("Available binaries:");
    eprintln!("  cargo run -p webgpu-sumcheck --bin u128_sumcheck -- <log2_len> [gpu_iters]");
    eprintln!("  cargo run -p webgpu-sumcheck --bin bn254_sumcheck -- [log2_len]");
    eprintln!("  cargo run -p webgpu-sumcheck --bin _bn254_mont32_test");
    eprintln!("  cargo run -p webgpu-sumcheck --bin _mul32_test");
    eprintln!();
    eprintln!("Available examples:");
    eprintln!("  cargo run -p webgpu-sumcheck --example gpu_vs_cpu_bn254 -- [len] [iters]");
    eprintln!("  cargo run -p webgpu-sumcheck --example gpu_vs_cpu_u32 -- [len] [iters]");
    eprintln!("  cargo run -p webgpu-sumcheck --example gpu_vs_cpu_u64 -- [len] [iters]");
    std::process::exit(1);
}
