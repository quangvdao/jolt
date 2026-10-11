//! Public bytecode H construction includes weight preparation and final table allocation.

use super::pipelines::BenchResult;
use super::runner::{measure_witness, summary, Options};
use super::witness::TraceFixture;
use common::constants::RAM_START_ADDRESS;
use jolt_field::F128;
use jolt_rv64i_prover::reference::bytecode::BytecodeReadAddressPrepare;
use jolt_rv64i_verifier::stages::stage6a::{BytecodeReadAddress, BytecodeReadPoints};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use rayon::ThreadPoolBuilder;
use std::error::Error;
use std::hint::black_box;
use std::sync::Arc;

#[expect(
    clippy::print_stdout,
    reason = "phase distributions are benchmark output"
)]
pub fn run(options: &Options) -> BenchResult<()> {
    let name = "bytecode_h";
    for &log_t in &options.log_t {
        if !options
            .threads
            .iter()
            .any(|&threads| options.selected_public(name, log_t, threads))
        {
            continue;
        }
        let fixture = TraceFixture::new(log_t)?;
        let bytecode = Arc::clone(fixture.preprocessing.shared_bytecode());
        drop(fixture);
        let mut transcript = Blake2bTranscript::<F128>::new(b"bytecode-setup");
        let x = transcript.challenge_vector(17);
        let points = BytecodeReadPoints::new(
            &x,
            transcript.challenge_vector(5),
            transcript.challenge_vector(usize::from(log_t)),
            transcript.challenge_vector(usize::from(log_t)),
            transcript.challenge_vector(usize::from(log_t)),
        )?;
        let relation = BytecodeReadAddress::new(
            bytecode.log_K(),
            points,
            RAM_START_ADDRESS,
            RAM_START_ADDRESS,
        )?;
        let challenges = relation.draw_challenges(&mut transcript)?;
        for &threads in &options.threads {
            if !options.selected_public(name, log_t, threads) {
                continue;
            }
            let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
            let _ = pool.broadcast(|_| black_box(()));
            let mut samples = Vec::with_capacity(options.samples);
            for _ in 0..options.samples {
                let (tables, sample) = pool
                    .install(|| {
                        measure_witness(name, 1 << log_t, options.inventory, || {
                            Ok(BytecodeReadAddressPrepare.public_tables(
                                &bytecode,
                                &relation,
                                &challenges,
                            )?)
                        })
                        .map_err(|error| error.to_string())
                    })
                    .map_err(|error| -> Box<dyn Error> { error.into() })?;
                drop(black_box(tables));
                samples.push(sample);
            }
            let cycles = (1_u64 << log_t) as f64;
            let (ns, min, max) = summary(
                samples
                    .iter()
                    .map(|sample| sample.elapsed.as_nanos() as f64 / cycles)
                    .collect(),
            );
            let peak = samples
                .iter()
                .map(|sample| sample.allocations.peak_bytes)
                .max()
                .unwrap_or(0);
            println!("witness_pipeline/{name}/{log_t}/{threads} samples={} ns_per_cycle={ns:.6} ms={:.6} min_ns={min:.6} min_ms={:.6} max_ns={max:.6} peak_bytes={peak} bytecode_rows={} loaded_machine=true", options.samples, ns * cycles / 1e6, min * cycles / 1e6, bytecode.rows().len());
        }
    }
    Ok(())
}
