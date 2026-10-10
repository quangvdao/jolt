//! Standalone lazy-column diagnostics; these are not a share of fused products.

use std::hint::black_box;
use std::time::Duration;

use jolt_field::F128;
use jolt_kernels::optimized::lazy_ra::{ChunkIndexSource, LazyFoldedRa, LazyRaError};
use jolt_rv64i_kernels::chunk_product::ChunkProductError;
use jolt_rv64i_kernels::par::{CycleChunks, ParError};
use jolt_rv64i_kernels::round::eq::eq_table;
use jolt_rv64i_kernels::source::{PrepareRequest, PresentGroup, SourceError, ValidatedTrace};
use jolt_rv64i_kernels::synth::SynthProfile;
use rayon::prelude::*;
use thiserror::Error;

use super::{run_prepared_cases, Case, Clock, RunnerError};

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
pub struct CompactDigits(PresentGroup);

impl ChunkIndexSource for CompactDigits {
    fn num_polys(&self) -> usize {
        self.0.num_polys()
    }
    fn cycles(&self) -> usize {
        self.0.cycles()
    }
    fn index(&self, column: usize, cycle: usize) -> Option<usize> {
        self.0.index(column, cycle)
    }
    fn index_bound(&self, column: usize) -> Option<usize> {
        self.0.index_bound(column)
    }
}

/// Keeps the final family resident until allocation measurement has ended.
pub struct GatherSample<S> {
    pub duration: Duration,
    pub columns: LazyFoldedRa<F128, S>,
}

/// Times canonical gathers, checksum reductions and the first four lazy binds,
/// including fourth-bind materialisation. Family construction is outside timing.
/// The diagnostic requires exactly five columns, matching both chunk groups.
pub fn measure_lazy_gathers<S: ChunkIndexSource>(
    source: S,
    tables: Vec<Vec<F128>>,
    challenges: &[F128],
) -> Result<GatherSample<S>, GatherError> {
    let log_t = source.cycles().ilog2() as usize;
    let d = source.num_polys();
    if d != 5 {
        return Err(ChunkProductError::Columns { columns: d }.into());
    }
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
                let mut values = [(F128::from_raw(0), F128::from_raw(0)); 5];
                let mut sum = F128::from_raw(0);
                for pair in start..end {
                    columns.lo_hi_all(pair, &mut values);
                    for &(low, high) in &values {
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
    let _ = run_prepared_cases(
        profiles,
        &cases,
        |source| {
            let (_, groups) = ValidatedTrace::prepare(
                source,
                PrepareRequest {
                    present: groups.iter().map(|(_, group)| group.clone()).collect(),
                    optional: vec![],
                },
            )?;
            Ok::<_, GatherError>(
                groups
                    .present
                    .into_iter()
                    .map(CompactDigits)
                    .collect::<Vec<_>>(),
            )
        },
        |sources, &index| Ok::<_, GatherError>(sources[index].clone()),
        |_, source, &index, challenges, times| {
            let tables = points(index)
                .iter()
                .map(|point| eq_table(point, None))
                .collect();
            let sample = measure_lazy_gathers(source, tables, challenges)?;
            times.set(0, sample.duration);
            let _ = black_box(&sample.columns);
            Ok::<_, GatherError>(sample)
        },
        |_, _| {},
        |record, _, _| {
            record.print_phase(&record.id, 0, None);
        },
    )?;
    Ok(())
}
