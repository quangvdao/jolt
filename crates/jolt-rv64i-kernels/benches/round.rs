//! Run with `RUSTFLAGS="-C target-cpu=native" cargo bench -p jolt-rv64i-kernels
//! --features test-utils --bench round`. Rows are median ns/call on hot inputs;
//! timings on a loaded machine are descriptive, not acceptance thresholds.

#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "standalone benchmark reports measurement rows on stdout"
)]

use jolt_field::F128;
use jolt_rv64i_kernels::round::{coefficients_from_nodes, quadratic};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::hint::black_box;
use std::time::Instant;

const BLOCK: usize = 1024;
const PASSES: usize = 32;
const SAMPLES: usize = 7;

struct Input {
    zero: F128,
    leading: F128,
    one: F128,
    nodes: [F128; 6],
}

fn main() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xa837);
    let inputs: Vec<_> = (0..BLOCK)
        .map(|_| {
            let mut field =
                || F128::from_raw((u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64()));
            Input {
                zero: field(),
                leading: field(),
                one: field(),
                nodes: std::array::from_fn(|_| field()),
            }
        })
        .collect();
    println!("kernel,degree,median_ns_per_call (loaded machine)");
    for degree in [3, 5, 8] {
        let run = || {
            for _ in 0..PASSES {
                for input in &inputs {
                    let input = black_box(input);
                    let _ = black_box(coefficients_from_nodes(
                        degree,
                        input.zero,
                        input.leading,
                        input.one,
                        &input.nodes[..degree - 2],
                    ));
                }
            }
        };
        run();
        let mut samples = [0.0; SAMPLES];
        for sample in &mut samples {
            let start = Instant::now();
            run();
            *sample = start.elapsed().as_nanos() as f64 / (BLOCK * PASSES) as f64;
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "coefficients_from_nodes,{degree},{:.2}",
            samples[SAMPLES / 2]
        );
    }
    let run = || {
        for _ in 0..PASSES {
            for input in &inputs {
                let input = black_box(input);
                let _ = black_box(quadratic(
                    [input.zero, input.leading],
                    [input.one, input.nodes[0]],
                ));
            }
        }
    };
    run();
    let mut samples = [0.0; SAMPLES];
    for sample in &mut samples {
        let start = Instant::now();
        run();
        *sample = start.elapsed().as_nanos() as f64 / (BLOCK * PASSES) as f64;
    }
    samples.sort_by(f64::total_cmp);
    println!("quadratic,2,{:.2}", samples[SAMPLES / 2]);
}
