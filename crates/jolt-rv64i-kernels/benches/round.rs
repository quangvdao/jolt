//! Run with `RUSTFLAGS="-C target-cpu=native" cargo bench -p jolt-rv64i-kernels
//! --features test-utils --bench round`. Rows are median ns/call on hot inputs;
//! timings on a loaded machine are descriptive, not acceptance thresholds.
//! For the native node-evaluation assembly, use the same flags with
//! `cargo rustc -p jolt-rv64i-kernels --features test-utils --release --bench round -- --emit=asm`.
//! The three node wrappers retain six-value and four-value callers for inspection.
//! Reconstruction timings include allocation and drop of the exact-length message vector.

#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "standalone benchmark reports measurement rows on stdout"
)]

use jolt_field::F128;
use jolt_rv64i_kernels::round::{
    coefficients_from_nodes, linear_at_nodes, quadratic, quadratic_at_nodes,
};
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
        let ns = median_ns(&inputs, |input| {
            let _ = black_box(coefficients_from_nodes(
                degree,
                input.zero,
                input.leading,
                input.one,
                &input.nodes[..degree - 2],
            ));
        });
        println!("coefficients_from_nodes,{degree},{ns:.2}");
    }
    let ns = median_ns(&inputs, |input| {
        let _ = black_box(quadratic(
            [input.zero, input.leading],
            [input.one, input.nodes[0]],
        ));
    });
    println!("quadratic,2,{ns:.2}");
    let ns = median_ns(&inputs, |input| {
        let _ = black_box(quadratic_nodes_six(black_box([
            input.zero,
            input.leading,
            input.one,
        ])));
    });
    println!("quadratic_at_nodes,6,{ns:.2}");
    let ns = median_ns(&inputs, |input| {
        let _ = black_box(quadratic_nodes_four(black_box([
            input.zero,
            input.leading,
            input.one,
        ])));
    });
    println!("quadratic_at_nodes,4,{ns:.2}");
    let ns = median_ns(&inputs, |input| {
        let _ = black_box(linear_nodes_six(black_box([input.zero, input.leading])));
    });
    println!("linear_at_nodes,6,{ns:.2}");
}

fn median_ns(inputs: &[Input], call: impl Fn(&Input)) -> f64 {
    let run = || {
        for _ in 0..PASSES {
            for input in inputs {
                call(black_box(input));
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
    samples[SAMPLES / 2]
}

#[inline(never)]
fn quadratic_nodes_six(q: [F128; 3]) -> [F128; 6] {
    quadratic_at_nodes(q)
}

#[inline(never)]
fn quadratic_nodes_four(q: [F128; 3]) -> [F128; 4] {
    let [a, b, c, d, _, _] = quadratic_at_nodes(q);
    [a, b, c, d]
}

#[inline(never)]
fn linear_nodes_six(l: [F128; 2]) -> [F128; 6] {
    linear_at_nodes(l)
}
