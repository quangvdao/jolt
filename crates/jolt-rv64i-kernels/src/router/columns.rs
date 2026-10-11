//! Routed witness columns from complete router fold tables.

use super::shape::{table_len, RouteEntry, RouterError, RouterShape};
use jolt_field::F128;

/// Returns `out[o] = Σ_ρ Σ_(o,s,h)∈route_ρ Fold_ρ[s,h]`.
/// Requires at least one shape, a common output domain and one complete fold apiece.
/// The caller supplies the folds of the intended trace and cycle point;
/// association with that trace is not checked here.
pub fn routed_columns(
    shapes: &[RouterShape],
    folds: &[Vec<F128>],
) -> Result<Vec<F128>, RouterError> {
    let first = shapes.first().ok_or(RouterError::EmptyShapes)?;
    if shapes.len() != folds.len() {
        return Err(RouterError::TableLength {
            table: "router folds",
            expected: shapes.len(),
            actual: folds.len(),
        });
    }
    let outputs = table_len(first.log_outputs())?;
    for (shape, fold) in shapes.iter().zip(folds) {
        if shape.log_outputs() != first.log_outputs() {
            return Err(RouterError::TableLength {
                table: "router output domain",
                expected: outputs,
                actual: table_len(shape.log_outputs())?,
            });
        }
        if fold.len() != shape.fold_len() {
            return Err(RouterError::TableLength {
                table: "router fold",
                expected: shape.fold_len(),
                actual: fold.len(),
            });
        }
    }
    let mut output = vec![F128::from_raw(0); outputs];
    for (shape, fold) in shapes.iter().zip(folds) {
        if shape.route().is_empty() {
            continue;
        }
        // fold_index owns the interleaved bit geometry. Source and selector
        // contributions occupy disjoint bits, checked by RouterShape::new.
        let sources = fold_indices(shape.bank().len() * u64::BITS as usize, |source| {
            shape.fold_index(source, 0)
        });
        let selectors = fold_indices(shape.selectors(), |selector| shape.fold_index(0, selector));
        let sources = sources.as_slice();
        let selectors = selectors.as_slice();
        let fold = fold.as_slice();
        // RouterShape::new checks power-of-two domains and route indices; the
        // fold-length check above pins the complete fold domain. These masks
        // therefore preserve every index while exposing its bounds to the loop.
        let source_mask = sources.len() - 1;
        let selector_mask = selectors.len() - 1;
        let fold_mask = fold.len() - 1;
        let value = |entry: &RouteEntry| {
            let index =
                sources[entry.source & source_mask] | selectors[entry.selector & selector_mask];
            fold[index & fold_mask]
        };
        // RouterShape::new sorts by output first, so each run has one destination.
        let mut destination = shape.route()[0].output;
        let mut sum = F128::from_raw(0);
        let mut add = |output_index, value| {
            if output_index != destination {
                output[destination] += sum;
                destination = output_index;
                sum = F128::from_raw(0);
            }
            sum += value;
        };
        let (blocks, remainder) = shape.route().as_chunks::<8>();
        for block in blocks {
            // RouterShape::new sorts the routes: equal endpoints make the whole
            // block one output run, permitting independent fold gathers.
            if block[0].output == block[7].output {
                add(
                    block[0].output,
                    value(&block[0])
                        + value(&block[1])
                        + value(&block[2])
                        + value(&block[3])
                        + value(&block[4])
                        + value(&block[5])
                        + value(&block[6])
                        + value(&block[7]),
                );
            } else {
                for entry in block {
                    add(entry.output, value(entry));
                }
            }
        }
        for entry in remainder {
            add(entry.output, value(entry));
        }
        output[destination] += sum;
    }
    Ok(output)
}

fn fold_indices(length: usize, basis: impl Fn(usize) -> usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(length);
    indices.push(0);
    // RouterShape::new checks distinct slots: fold_index deposits disjoint bits.
    for bit in 0..length.ilog2() {
        let image = basis(1 << bit);
        for index in 0..indices.len() {
            indices.push(indices[index] | image);
        }
    }
    indices
}
