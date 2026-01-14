use std::time::{Duration, Instant};
use tracing::info;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

// Empirically measured cycles per SHA256 operation (from e2e_profiling.rs)
const CYCLES_PER_SHA256: f64 = 3396.0;
const SAFETY_MARGIN: f64 = 0.9; // Use 90% of max trace capacity

/// Calculate number of SHA2 iterations to target a specific cycle count
fn scale_to_iters(scale: usize) -> u32 {
    let target_cycles = ((1usize << scale) as f64 * SAFETY_MARGIN) as usize;
    std::cmp::max(1, (target_cycles as f64 / CYCLES_PER_SHA256) as u32)
}

/// Simple statistics for benchmark runs
struct BenchStats {
    times: Vec<Duration>,
}

impl BenchStats {
    fn new() -> Self {
        Self { times: Vec::new() }
    }

    fn record(&mut self, duration: Duration) {
        self.times.push(duration);
    }

    fn mean(&self) -> Duration {
        let total: Duration = self.times.iter().sum();
        total / self.times.len() as u32
    }

    fn min(&self) -> Duration {
        *self.times.iter().min().unwrap()
    }

    fn max(&self) -> Duration {
        *self.times.iter().max().unwrap()
    }

    fn std_dev(&self) -> f64 {
        let mean_secs = self.mean().as_secs_f64();
        let variance: f64 = self
            .times
            .iter()
            .map(|t| {
                let diff = t.as_secs_f64() - mean_secs;
                diff * diff
            })
            .sum::<f64>()
            / self.times.len() as f64;
        variance.sqrt()
    }

    fn report(&self, name: &str) {
        println!("\n========== {} ==========", name);
        println!("  Runs:    {}", self.times.len());
        println!("  Mean:    {:.3} s", self.mean().as_secs_f64());
        println!("  Std Dev: {:.3} s", self.std_dev());
        println!("  Min:     {:.3} s", self.min().as_secs_f64());
        println!("  Max:     {:.3} s", self.max().as_secs_f64());
        println!(
            "  All runs: {:?}",
            self.times
                .iter()
                .map(|d| format!("{:.3}s", d.as_secs_f64()))
                .collect::<Vec<_>>()
        );
    }
}

/// Setup tracing with optional Chrome trace output
fn setup_tracing(enable_chrome: bool, trace_name: &str) -> Option<tracing_chrome::FlushGuard> {
    if enable_chrome {
        let trace_dir = "benchmark-runs/traces";
        std::fs::create_dir_all(trace_dir).ok();
        let trace_path = format!("{}/{}.json", trace_dir, trace_name);

        let (chrome_layer, guard) = ChromeLayerBuilder::new()
            .file(&trace_path)
            .include_args(true)
            .build();

        tracing_subscriber::registry()
            .with(chrome_layer)
            .with(tracing_subscriber::fmt::layer().with_filter(
                tracing_subscriber::EnvFilter::from_default_env()
            ))
            .init();

        println!(">>> Chrome tracing enabled, will write to: {}", trace_path);
        Some(guard)
    } else {
        tracing_subscriber::fmt::init();
        None
    }
}

pub fn main() {
    // Configuration
    let num_runs: usize = std::env::var("BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let enable_trace = std::env::var("TRACE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    // Determine iterations: either from SCALE (power of 2) or direct SHA2_ITERS
    let (iters, scale, scale_info) = if let Some(scale) = std::env::var("SCALE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        let iters = scale_to_iters(scale);
        let target_cycles = ((1usize << scale) as f64 * SAFETY_MARGIN) as usize;
        (
            iters,
            scale,
            format!(
                "2^{} (~{:.1}M target cycles, {:.0}% of 2^{})",
                scale,
                target_cycles as f64 / 1_000_000.0,
                SAFETY_MARGIN * 100.0,
                scale
            ),
        )
    } else if let Some(iters) = std::env::var("SHA2_ITERS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
    {
        // Estimate scale from iters
        let est_cycles = iters as f64 * CYCLES_PER_SHA256;
        let est_scale = (est_cycles.log2().ceil() as usize).max(16);
        (iters, est_scale, format!("manual ({} iters)", iters))
    } else {
        // Default to scale 22 (similar to guest's max_trace_length = 4194304 = 2^22)
        let default_scale = 22;
        let iters = scale_to_iters(default_scale);
        (iters, default_scale, format!("default 2^{}", default_scale))
    };

    // Setup tracing
    let timestamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let trace_name = format!("sha2_chain_scale{}_{}", scale, timestamp);
    let _trace_guard = setup_tracing(enable_trace, &trace_name);

    println!("=== SHA2-Chain Benchmark ===");
    println!("  Scale:           {}", scale_info);
    println!("  SHA2 iterations: {}", iters);
    println!(
        "  Est. cycles:     {:.2}M",
        iters as f64 * CYCLES_PER_SHA256 / 1_000_000.0
    );
    println!("  Benchmark runs:  {}", num_runs);
    println!("  Chrome tracing:  {}", if enable_trace { "enabled" } else { "disabled" });

    // ========== PREPROCESSING (done once) ==========
    println!("\n>>> Preprocessing (one-time)...");
    let preprocess_start = Instant::now();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_sha2_chain(target_dir);

    let shared_preprocessing = guest::preprocess_shared_sha2_chain(&mut program);
    let prover_preprocessing = guest::preprocess_prover_sha2_chain(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_sha2_chain(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
    );

    let prove_sha2_chain = guest::build_prover_sha2_chain(program, prover_preprocessing);
    let verify_sha2_chain = guest::build_verifier_sha2_chain(verifier_preprocessing);

    let preprocess_time = preprocess_start.elapsed();
    println!(
        ">>> Preprocessing done in {:.3} s",
        preprocess_time.as_secs_f64()
    );

    // ========== BENCHMARK PROVING ==========
    let input = [5u8; 32];
    let native_output = guest::sha2_chain(input, iters);

    let mut prove_stats = BenchStats::new();

    for run in 1..=num_runs {
        println!("\n--- Run {}/{} ---", run, num_runs);

        let prove_start = Instant::now();
        let (output, proof, program_io) = prove_sha2_chain(input, iters);
        let prove_time = prove_start.elapsed();

        prove_stats.record(prove_time);
        info!("  Prove time: {:.3} s", prove_time.as_secs_f64());

        // Verify correctness (at least on first run)
        if run == 1 {
            assert_eq!(output, native_output, "output mismatch");
            let is_valid = verify_sha2_chain(input, iters, output, program_io.panic, proof);
            assert!(is_valid, "proof verification failed");
            info!("  Verification: PASSED");
        }
    }

    // ========== REPORT ==========
    prove_stats.report("Proving Time");
    println!(
        "\nPreprocessing time (one-time): {:.3} s",
        preprocess_time.as_secs_f64()
    );

    if enable_trace {
        println!("\n>>> Trace file saved. Run analysis with:");
        println!("    python3 scripts/analyze_trace.py benchmark-runs/traces/{}.json", trace_name);
    }
}
