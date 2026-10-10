//! Compare the digit builder and shared-table reduction on local traces.

pub mod support;

use std::fmt::Display;
use std::hint::black_box;
use std::sync::Arc;

use jolt_field::F128;
use jolt_rv64i_kernels::packed::lift::WordLift;
use jolt_rv64i_kernels::par::CycleChunks;
use jolt_rv64i_kernels::reduction::{g_pass_digits, ReductionCore, ReductionError, ReductionLeg};
use jolt_rv64i_kernels::source::{CycleSource, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_utils::unsafe_allocate_zero_vec;
use rayon::prelude::*;

use support::{core_rounds, run_cases, Case, Clock, RunnerError};

const ZERO: F128 = F128::from_raw(0);
const ONE: F128 = F128::from_raw(1);

struct Fixture {
    source: Arc<SyntheticTrace>,
    validated: ValidatedTrace<SyntheticTrace>,
}

fn weights() -> Vec<Vec<F128>> {
    (0..3)
        .map(|support| {
            (0..256)
                .map(|column| {
                    let active = match support {
                        0 => (64..=228).contains(&column),
                        1 => column < 64 || (139..=230).contains(&column),
                        _ => column < 64,
                    };
                    if active {
                        let scalar = if column < 64 { 9 } else { support as u128 + 7 };
                        F128::from_raw(((column as u128 + 1) << 64) | scalar)
                    } else {
                        ZERO
                    }
                })
                .collect()
        })
        .collect()
}

#[expect(
    clippy::expect_used,
    reason = "SyntheticTrace construction checks the cycle exponent"
)]
fn word_table(source: &SyntheticTrace, weight: &[F128]) -> Vec<F128> {
    let mut values = [ZERO; 64];
    values.copy_from_slice(&weight[..64]);
    let lift = WordLift::new(&values);
    let mut table = unsafe_allocate_zero_vec(source.cycles());
    let chunks = CycleChunks::new(source.cycles().ilog2() as usize, 0)
        .expect("synthetic trace has validated geometry");
    table
        .par_chunks_mut(chunks.chunk_len())
        .enumerate()
        .for_each(|(chunk, output)| {
            for (offset, value) in output.iter_mut().enumerate() {
                *value = lift.lift(source.trace_word(5, chunk * chunks.chunk_len() + offset));
            }
        });
    table
}

fn core_from_tables(
    tables: Vec<Vec<F128>>,
    cycles: usize,
    shared: bool,
) -> Result<(ReductionCore, F128), ReductionError> {
    let log_t = cycles.ilog2() as usize;
    let count = if shared { 4 } else { 3 };
    let mut legs = Vec::with_capacity(count);
    let mut claim = ZERO;
    for leg in 0..count {
        let table = if shared { [0, 1, 2, 2][leg] } else { leg };
        // Boolean points give honest input claims by one table read, keeping
        // fixture summation out of the model's measured construction pass.
        let point_leg = if shared { [0, 1, 1, 2][leg] } else { leg };
        let vertex = (0x39a5usize.wrapping_mul(point_leg + 1)) & (cycles - 1);
        let point = (0..log_t)
            .map(|bit| if vertex & (1 << bit) == 0 { ZERO } else { ONE })
            .collect();
        let coefficient = F128::from_raw(if shared {
            [1, 2, 2, 3][leg]
        } else {
            leg as u128 + 1
        });
        let value = tables[table][vertex];
        claim += coefficient * value;
        legs.push(ReductionLeg {
            table,
            point,
            coefficient,
            claim: value,
        });
    }
    Ok((ReductionCore::new(tables, legs)?, claim))
}

fn bench_error(error: impl Display) -> RunnerError {
    RunnerError::Core {
        message: error.to_string(),
    }
}

#[expect(
    clippy::print_stdout,
    reason = "prepared-table ownership is benchmark output"
)]
fn main() -> Result<(), RunnerError> {
    let map = SyntheticTrace::column_map();
    let weights = weights();
    let mut pass_weights = weights[..2].to_vec();
    pass_weights[1][..64].fill(ZERO);
    let mut default = Case::core("reduction", false, &["prepare"]);
    default.threshold = Some(|_, _| 30.0);
    let cases = [default, Case::core("reduction_shared", true, &["prepare"])];
    let _ = run_cases(
        &[SynthProfile::Local],
        &cases,
        |source| {
            Ok::<_, RunnerError>(Fixture {
                validated: ValidatedTrace::new(Arc::clone(&source)).map_err(bench_error)?,
                source,
            })
        },
        |fixture, &shared, challenges, times| {
            // This owned table is transferred into the core, so its untimed
            // preparation remains inside each sample's allocation interval.
            let prepared = if shared {
                let clock = Clock::start();
                let table = word_table(&fixture.source, &weights[2]);
                times.set(4, clock.elapsed());
                Some(table)
            } else {
                None
            };
            let clock = Clock::start();
            let mut tables = g_pass_digits(
                &fixture.validated,
                &map,
                if shared { &pass_weights } else { &weights },
            )
            .map_err(bench_error)?;
            if let Some(prepared) = prepared {
                tables.push(prepared);
            }
            let (mut core, claim) =
                core_from_tables(tables, fixture.source.cycles(), shared).map_err(bench_error)?;
            times.set(0, clock.elapsed());
            let rounds = core_rounds(&mut core, claim, challenges)?;
            times.set(1, rounds.rounds);
            times.set(2, rounds.finish);
            let clock = Clock::start();
            let values = core.final_values().map_err(bench_error)?.to_vec();
            let _ = black_box(&values);
            times.set(3, clock.elapsed());
            Ok((core, values))
        },
        |_, _, _| {},
        |record, _, &shared| {
            if shared {
                println!("{}_prepared prepared_bytes={} incremental_peak_bytes={} incremental_final_bytes={} allocs_with_preparation={} loaded_machine=true", record.id,
                    (1usize << record.log_t) * std::mem::size_of::<F128>(),
                    record.allocation.peak_bytes, record.allocation.final_bytes, record.allocation.allocs);
            }
        },
    )?;
    Ok(())
}
