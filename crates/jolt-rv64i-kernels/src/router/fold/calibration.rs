//! Checked experiment geometry for machinery and unit-cost benchmarks.

use super::{fold_impl, FoldLayout, FoldOutput, PhaseHook, ShapeLayout};
use crate::packed::scatter::ScatterPlan;
use crate::router::shape::{synthetic_router_shapes, RouterError, RouterShape};
use crate::source::{CycleSource, ValidatedTrace};
use crate::synth::{SynthProfile, SyntheticTrace};
use jolt_field::F128;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Geometry of the five synthetic router shapes, derived by `FoldLayout::new`.
/// Available only with `test-utils`; it owns no production routing state.
#[derive(Debug, Clone)]
pub struct FoldCalibration {
    layout: FoldLayout,
    selectors: [usize; 5],
    word_sets: [usize; 5],
    offsets: [usize; 5],
}

impl FoldCalibration {
    /// Uses the first `byte_selectors` Variant values as byte buckets, clamped
    /// to that shape's selector count. Source indices and all storage ranges
    /// are checked by `ValidatedTrace::new` and `FoldLayout::new`.
    pub fn new(byte_selectors: usize) -> Result<Self, RouterError> {
        let shapes = synthetic_router_shapes()?;
        let trace = SyntheticTrace::new(SynthProfile::AllRows, 1, 1, 0x5eed)
            .map_err(|_| RouterError::Dimension { variables: 1 })?;
        let trace = ValidatedTrace::new(Arc::new(trace))
            .map_err(|_| RouterError::Dimension { variables: 1 })?;
        let byte_values: Vec<Vec<usize>> = shapes
            .iter()
            .enumerate()
            .map(|(shape, layout)| {
                if shape == 0 {
                    (0..byte_selectors.min(layout.selectors())).collect()
                } else {
                    Vec::new()
                }
            })
            .collect();
        let layout = FoldLayout::new(&trace, &shapes, &byte_values)?;
        let checked: &[ShapeLayout; 5] =
            layout
                .shapes
                .as_slice()
                .try_into()
                .map_err(|_| RouterError::TableLength {
                    table: "calibration shapes",
                    expected: 5,
                    actual: layout.shapes.len(),
                })?;
        let selectors = std::array::from_fn(|shape| checked[shape].selectors);
        let word_sets = std::array::from_fn(|shape| checked[shape].words.len());
        let offsets = std::array::from_fn(|shape| {
            if shape == 0 {
                checked[shape].metadata
            } else {
                checked[shape]
                    .bases
                    .first()
                    .copied()
                    .unwrap_or(checked[shape].metadata)
            }
        });
        Ok(Self {
            layout,
            selectors,
            word_sets,
            offsets,
        })
    }

    /// Cycle scratch elements, including digit and flag metadata.
    pub fn entries(&self) -> usize {
        self.layout.entries()
    }

    /// Row scratch elements for the separate visited-bytecode-row pass.
    pub fn row_entries(&self) -> usize {
        self.layout.row_entries()
    }

    /// Variant metadata start followed by the four other cycle-shape starts.
    pub fn offsets(&self) -> [usize; 5] {
        self.offsets
    }

    /// Selector-domain sizes in synthetic router shape order.
    pub fn selectors(&self) -> [usize; 5] {
        self.selectors
    }

    /// Cycle-bucketed word counts in synthetic router shape order.
    pub fn word_sets(&self) -> [usize; 5] {
        self.word_sets
    }

    /// The Variant word span for a valid selector; invalid values return `Layout`.
    pub fn variant_base(&self, selector: usize) -> Result<usize, RouterError> {
        self.shape_base(0, selector)
    }

    /// A cycle word span. Invalid shapes return `SlotRange`; invalid selectors
    /// return `Layout`. Both indices are checked before reading storage metadata.
    pub fn shape_base(&self, shape: usize, selector: usize) -> Result<usize, RouterError> {
        let layout = self
            .layout
            .shapes
            .get(shape)
            .ok_or(RouterError::SlotRange {
                slot: shape,
                slots: self.layout.shapes.len(),
            })?;
        layout
            .bases
            .get(selector)
            .copied()
            .ok_or(RouterError::Layout {
                shape,
                selector,
                bound: layout.selectors,
            })
    }

    /// The Variant digit/flag span; invalid selector values return `Layout`.
    pub fn variant_metadata_base(&self, selector: usize) -> Result<usize, RouterError> {
        let layout = self
            .layout
            .shapes
            .first()
            .ok_or(RouterError::SlotRange { slot: 0, slots: 0 })?;
        if selector >= layout.selectors {
            return Err(RouterError::Layout {
                shape: 0,
                selector,
                bound: layout.selectors,
            });
        }
        Ok(layout.metadata + selector * layout.meta_len)
    }
}

struct PhaseClock {
    times: [Duration; 4],
    start: Instant,
}

impl PhaseClock {
    fn new() -> Self {
        Self {
            times: [Duration::ZERO; 4],
            start: Instant::now(),
        }
    }
}

impl PhaseHook for PhaseClock {
    fn finish_phase(&mut self, phase: usize) {
        self.times[phase] += self.start.elapsed();
        self.start = Instant::now();
    }
}

impl FoldLayout {
    /// Benchmark-only observation of the identical fold implementation. Durations
    /// cover fused equality/buckets/emission, scatter application, visited-row
    /// buckets, and preparation/merges/read-out, respectively. Lazy scratch
    /// zero-fill is included in its bucket phase; the first phase cannot isolate
    /// its fused multiplication and XORs without instrumenting every cycle.
    pub fn measure<S: CycleSource>(
        &self,
        source: &ValidatedTrace<S>,
        shapes: &[RouterShape],
        point: &[F128],
        plan: &ScatterPlan<S>,
        histogram_columns: &[usize],
    ) -> Result<(FoldOutput, [Duration; 4]), RouterError> {
        let mut phases = PhaseClock::new();
        let output = fold_impl(
            source,
            shapes,
            point,
            plan,
            self,
            histogram_columns,
            &mut phases,
        )?;
        Ok((output, phases.times))
    }
}
