//! Compare the digit builder, byte builder and shared-table reduction on local traces.

pub mod support;

use std::sync::{Arc, Mutex};

use jolt_field::F128;
use jolt_rv64i_kernels::packed::lift::WordLift;
use jolt_rv64i_kernels::par::CycleChunks;
use jolt_rv64i_kernels::reduction::{
    g_pass_bytes, g_pass_digits, ColumnMap, ReductionCore, ReductionError,
};
use jolt_rv64i_kernels::source::{CycleSource, SourceError, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use rayon::prelude::*;
use thiserror::Error;

use support::{run_core, RunnerError};

const ZERO: F128 = F128::from_raw(0);
const ONE: F128 = F128::from_raw(1);

#[derive(Debug, Error)]
enum BenchError {
    #[error(transparent)]
    Source(#[from] SourceError),
    #[error(transparent)]
    Reduction(#[from] ReductionError),
    #[error("validated fixture cache lock was poisoned")]
    Cache,
}

struct Fixture {
    source: Arc<SyntheticTrace>,
    validated: ValidatedTrace<SyntheticTrace>,
    word: Vec<F128>,
}

fn map() -> Vec<ColumnMap> {
    let mut map = vec![ColumnMap::Word {
        start: 0,
        trace_word: 5,
    }];
    for column in 0..10 {
        map.push(ColumnMap::Indicators {
            start: 64 + 15 * column,
            column,
        });
    }
    for (start, column) in [(214, 10), (221, 11)] {
        map.push(ColumnMap::Indicators { start, column });
    }
    map.push(ColumnMap::Flags {
        start: 228,
        columns: vec![18, 19, 20],
    });
    map
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
    let mut table = vec![ZERO; source.cycles()];
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

fn run_variant(name: &str, bytes: bool, shared: bool) -> Result<(), RunnerError> {
    let cache = Mutex::new(None::<Fixture>);
    let map = map();
    let weights = weights();
    let pass_weights = if shared {
        let mut weights = weights[..2].to_vec();
        weights[1][..64].fill(ZERO);
        weights
    } else {
        weights.clone()
    };
    run_core(
        name,
        &[SynthProfile::Local],
        |source| {
            let tables = if bytes {
                g_pass_bytes(source.rows(), &pass_weights)?
            } else {
                let mut cached = cache.lock().map_err(|_| BenchError::Cache)?;
                if cached
                    .as_ref()
                    .is_none_or(|fixture| !Arc::ptr_eq(&fixture.source, &source))
                {
                    *cached = Some(Fixture {
                        validated: ValidatedTrace::new(Arc::clone(&source))?,
                        word: if shared {
                            word_table(&source, &weights[2])
                        } else {
                            Vec::new()
                        },
                        source: Arc::clone(&source),
                    });
                }
                let fixture = cached.as_ref().ok_or(BenchError::Cache)?;
                let mut tables = g_pass_digits(&fixture.validated, &map, &pass_weights)?;
                if shared {
                    tables.push(fixture.word.clone());
                }
                tables
            };
            let log_t = source.cycles().ilog2() as usize;
            let count = if shared { 4 } else { 3 };
            let mut legs = Vec::with_capacity(count);
            let mut claim = ZERO;
            for leg in 0..count {
                let table = if shared { [0, 1, 2, 2][leg] } else { leg };
                // Boolean points give honest input claims by one table read, keeping
                // fixture summation out of the model's measured construction pass.
                let point_leg = if shared { [0, 1, 1, 2][leg] } else { leg };
                let vertex = (0x39a5usize.wrapping_mul(point_leg + 1)) & (source.cycles() - 1);
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
                legs.push((table, point, coefficient, value));
            }
            let core = ReductionCore::new(tables, legs)?;
            Ok::<_, BenchError>((core, claim))
        },
        |core, _| Ok::<_, BenchError>(core.final_values()?.to_vec()),
    )
}

fn main() -> Result<(), RunnerError> {
    run_variant("reduction", false, false)?;
    run_variant("reduction_bytes", true, false)?;
    run_variant("reduction_shared", false, true)
}
