//! Standalone lazy-column diagnostics; these are not a share of fused products.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use jolt_field::F128;
use jolt_kernels::optimized::lazy_ra::{ChunkIndexSource, LazyFoldedRa, LazyRaError};
use jolt_rv64i_kernels::chunk_product::ChunkProductError;
use jolt_rv64i_kernels::par::{CycleChunks, ParError};
use jolt_rv64i_kernels::round::eq::eq_table;
use jolt_rv64i_kernels::source::{CycleSource, DigitColumns, SourceError, ValidatedTrace};
use jolt_rv64i_kernels::synth::SynthProfile;
use jolt_utils::unsafe_allocate_zero_vec;
use rayon::prelude::*;
use thiserror::Error;

use super::{run_cases, Case, Clock, RunnerError};

#[derive(Debug, Error)]
pub enum GatherError {
    #[error(transparent)]
    Source(#[from] SourceError),
    #[error(transparent)]
    Chunk(#[from] ChunkProductError),
    #[error(transparent)]
    Par(#[from] ParError),
    #[error(transparent)]
    Lazy(#[from] LazyRaError),
}

/// Present row-major digits, validated and compacted outside the gather timer.
#[derive(Clone)]
pub struct CompactDigits {
    digits: Vec<u8>,
    bits: [usize; 7],
    d: usize,
    cycles: usize,
}

impl CompactDigits {
    pub fn new<S: CycleSource>(columns: &DigitColumns<S>) -> Result<Self, GatherError> {
        let cycles = columns.cycles();
        let d = columns.num_polys();
        if !(1..=7).contains(&d) {
            return Err(ChunkProductError::Columns { columns: d }.into());
        }
        let bits = std::array::from_fn(|column| {
            if column < d {
                columns.source().bits(columns.columns()[column])
            } else {
                0
            }
        });
        for (column, &bits) in bits.iter().take(d).enumerate() {
            if bits > 8 {
                return Err(ChunkProductError::ColumnWidth { column, bits }.into());
            }
        }
        let geometry = CycleChunks::new(cycles.ilog2() as usize, 0)?;
        let mut digits = unsafe_allocate_zero_vec(cycles * d);
        let missing = digits
            .par_chunks_mut(geometry.chunk_len() * d)
            .enumerate()
            .find_map_first(|(chunk, digits)| {
                let start = chunk * geometry.chunk_len();
                for (offset, row) in digits.chunks_exact_mut(d).enumerate() {
                    for (column, digit) in row.iter_mut().enumerate() {
                        let cycle = start + offset;
                        match columns.index(column, cycle) {
                            Some(index) => *digit = index as u8,
                            None => return Some(ChunkProductError::MissingDigit { column, cycle }),
                        }
                    }
                }
                None
            });
        if let Some(error) = missing {
            return Err(error.into());
        }
        Ok(Self {
            digits,
            bits,
            d,
            cycles,
        })
    }
}

impl ChunkIndexSource for CompactDigits {
    fn num_polys(&self) -> usize {
        self.d
    }
    fn cycles(&self) -> usize {
        self.cycles
    }
    fn index(&self, column: usize, cycle: usize) -> Option<usize> {
        Some(usize::from(self.digits[cycle * self.d + column]))
    }
    fn index_bound(&self, column: usize) -> Option<usize> {
        Some(1 << self.bits[column])
    }
}

/// Keeps the final family resident until allocation measurement has ended.
pub struct GatherSample<S> {
    pub duration: Duration,
    pub columns: LazyFoldedRa<F128, S>,
}

/// Times canonical gathers, checksum reductions and the first four lazy binds,
/// including fourth-bind materialisation. Family construction is outside timing.
pub fn measure_lazy_gathers<S: ChunkIndexSource>(
    source: S,
    tables: Vec<Vec<F128>>,
    challenges: &[F128],
) -> Result<GatherSample<S>, GatherError> {
    let log_t = source.cycles().ilog2() as usize;
    let d = source.num_polys();
    let mut columns = LazyFoldedRa::try_new(tables, source)?;
    let mut duration = Duration::ZERO;
    for (round, &challenge) in challenges.iter().take(log_t.min(4)).enumerate() {
        let geometry = CycleChunks::new(log_t, round)?;
        let clock = Clock::start();
        let checksum = (0..geometry.len() / geometry.chunk_len())
            .into_par_iter()
            .map(|chunk| {
                let start = chunk * geometry.chunk_len() / 2;
                let end = start + geometry.chunk_len() / 2;
                let mut values = [(F128::from_raw(0), F128::from_raw(0)); 7];
                let mut sum = F128::from_raw(0);
                for pair in start..end {
                    columns.lo_hi_all(pair, &mut values[..d]);
                    for &(low, high) in &values[..d] {
                        sum += low + high;
                    }
                }
                sum
            })
            .reduce(|| F128::from_raw(0), |left, right| left + right);
        let _ = black_box(checksum);
        columns.bind(challenge);
        duration += clock.elapsed();
    }
    Ok(GatherSample { duration, columns })
}

/// Runner entry for independent five-column diagnostics on a synthetic profile.
pub fn run_gathers(
    profiles: &[SynthProfile],
    groups: &[(&str, Vec<usize>)],
    points: impl Fn(usize) -> Vec<Vec<F128>> + Sync,
) -> Result<(), RunnerError> {
    let cases: Vec<_> = groups
        .iter()
        .enumerate()
        .map(|(index, (name, _))| Case {
            name: (*name).to_owned(),
            variant: index,
            phases: vec!["standalone_gathers_and_lazy_binds".to_owned()],
            total: vec![0],
            threshold: None,
        })
        .collect();
    let _ = run_cases(
        profiles,
        &cases,
        |source| {
            let trace = Arc::new(ValidatedTrace::new(source)?);
            groups
                .iter()
                .map(|(_, group)| {
                    let columns = DigitColumns::from_validated(Arc::clone(&trace), group.clone())?;
                    CompactDigits::new(&columns)
                })
                .collect::<Result<Vec<_>, GatherError>>()
        },
        |sources, &index, challenges, times| {
            let tables = points(index)
                .iter()
                .map(|point| eq_table(point, None))
                .collect();
            let sample = measure_lazy_gathers(sources[index].clone(), tables, challenges)?;
            times.set(0, sample.duration);
            let _ = black_box(&sample.columns);
            Ok::<_, GatherError>(sample)
        },
        |_, _, _| {},
        |record, _, _| {
            record.print_phase(&record.id, 0, None);
        },
    )?;
    Ok(())
}
