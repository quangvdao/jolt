//! Source abstractions for commitment backends.
//!
//! A source describes committed data and the traversal shapes a backend may
//! exploit. It does not prescribe the backend's commitment algorithm or
//! parallel schedule.

use jolt_field::Field;
use jolt_poly::{MultilinearPoly, OneHotPolynomial, Polynomial, RlcSource};

/// Stable identifier for a committed source inside a batch commitment source.
///
/// In the Dory/Jolt trace path this can be a logical committed polynomial id.
/// In a packed PCS path this can instead identify a packed witness group. The
/// id names what the PCS commits to; it does not have to be one logical Jolt
/// polynomial.
pub trait SourceId: Copy + Eq + Ord + Send + Sync + 'static {}

impl<T> SourceId for T where T: Copy + Eq + Ord + Send + Sync + 'static {}

/// A compact coordinate into a one-hot domain.
///
/// The value is the hot basis-vector index `k` in `e_k`. The surrounding
/// [`OneHotRow`] carries the domain size, so this type only stores the
/// coordinate. Current Jolt one-hot chunks have at most `2^8` entries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct OneHotIndex(u8);

impl OneHotIndex {
    /// Creates a one-hot coordinate when `index < 2^log_domain_size`.
    pub fn new(index: u8, log_domain_size: u8) -> Option<Self> {
        (log_domain_size <= 8 && (index as usize) < (1usize << log_domain_size))
            .then_some(Self(index))
    }

    /// Returns the coordinate as an array/vector index.
    pub fn get(self) -> usize {
        self.0 as usize
    }
}

/// A row of one-hot entries, one entry per trace column in the current chunk.
///
/// `log_domain_size` says that every hot coordinate lives in a one-hot domain
/// of size `2^log_domain_size`. The entries record whether each trace column
/// has a required hot coordinate or may be zero. Dory consumes this as the
/// current streaming one-hot chunk shape: it builds one row commitment per hot
/// coordinate, with columns contributing to the row for their hot coordinate.
pub struct OneHotRow<'a> {
    pub log_domain_size: u8,
    pub entries: OneHotEntries<'a>,
}

/// Per-column one-hot data for a [`OneHotRow`].
///
/// This enum avoids forcing all one-hot rows through `Option`. Rows such as
/// instruction and bytecode RA have one hot coordinate for every trace column.
/// Rows such as RAM RA can have no committed address for a column after address
/// remapping, so they need the zero-or-one representation.
pub enum OneHotEntries<'a> {
    /// Every trace column contributes exactly one one-hot basis vector.
    ///
    /// Entry `indices[col] = k` means column `col` contributes `e_k`.
    OnePerColumn(&'a [OneHotIndex]),

    /// Each trace column contributes either zero or one one-hot basis vector.
    ///
    /// Entry `indices[col] = Some(k)` means column `col` contributes `e_k`.
    /// Entry `indices[col] = None` means column `col` contributes zero.
    MaybeZero(&'a [Option<OneHotIndex>]),
}

/// A borrowed row view of a polynomial source.
///
/// This is a traversal hint, not the core polynomial abstraction. Backends that
/// can exploit row structure consume these rows directly. Backends that do not
/// care about the encoding may interpret the row as field evaluations.
pub enum SourceRow<'a, F> {
    /// A dense row of field evaluations.
    FieldElements(&'a [F]),

    /// A dense row of signed integers embedded canonically into the field.
    ///
    /// This preserves small-scalar MSM paths without first materializing field
    /// elements.
    I128(&'a [i128]),

    /// A dense row of unsigned 64-bit integers embedded canonically into the field.
    ///
    /// This preserves the common compact-polynomial benchmark and commitment
    /// path without paying the cost of first converting every row entry into a
    /// full-width field element.
    U64(&'a [u64]),

    /// A streaming one-hot chunk whose entries are one-hot vectors over a small
    /// domain.
    ///
    /// This is included for Jolt's RA commitments, where Dory can preserve the
    /// existing grouped-addition path without materializing a dense `{0,1}`
    /// table. Backends that do not exploit this shape can expand it explicitly
    /// in the same hot-coordinate-major order.
    OneHot(OneHotRow<'a>),
}

/// A single polynomial-like object that a PCS can commit to and open.
///
/// The source owns semantic operations: evaluate at a point, traverse rows, and
/// fold rows for opening-time vector/matrix products. It may be materialized or
/// lazy; for example, it can be backed by an execution trace.
pub trait CommitmentSource<F: Field>: Send + Sync {
    /// Number of multilinear variables in the source.
    fn num_vars(&self) -> usize;

    /// Evaluates the source at a multilinear point.
    fn evaluate(&self, point: &[F]) -> F;

    /// Visits row-shaped chunks of the source using `sigma` column variables.
    ///
    /// The borrowed row only has to remain valid for the duration of the visit
    /// call, which lets trace-backed sources allocate temporary row buffers and
    /// avoid ownership wrappers such as `Cow`.
    fn for_each_row<V>(&self, sigma: usize, visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, F>);

    /// Maps row-shaped chunks of the source into owned backend results.
    ///
    /// This is the performance-oriented companion to
    /// [`for_each_row`](Self::for_each_row). The default implementation is a
    /// sequential traversal, which is sufficient for lazy sources that produce
    /// temporary row buffers. Materialized sources can override this method to
    /// parallelize over borrowed row chunks without first copying them into an
    /// owned staging buffer.
    fn map_rows<R, V>(&self, sigma: usize, visit: V) -> Vec<R>
    where
        R: Send,
        V: for<'row> Fn(usize, SourceRow<'row, F>) -> R + Send + Sync,
    {
        let mut rows = Vec::new();
        self.for_each_row(sigma, |row_index, row| {
            rows.push(visit(row_index, row));
        });
        rows
    }

    /// Whether the whole source is a unit-valued one-hot polynomial.
    ///
    /// This preserves existing commitment fast paths that only need hot
    /// coordinates instead of row materialization. Sources that expose richer
    /// per-row one-hot structure can use [`SourceRow::OneHot`] instead.
    fn is_one_hot(&self) -> bool {
        false
    }

    /// Visits the hot flat indices when [`is_one_hot`](Self::is_one_hot) is true.
    ///
    /// Backends use this for current Dory-style one-hot commitment, where each
    /// hot index maps directly to one SRS basis addition. Non-one-hot sources
    /// may leave the default empty traversal.
    fn for_each_one<V>(&self, _visit: V)
    where
        V: FnMut(usize),
    {
    }

    /// Folds rows against the left-side weights used by opening algorithms.
    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F>;
}

fn multilinear_num_vars<F, T>(source: &T) -> usize
where
    F: Field,
    T: MultilinearPoly<F> + ?Sized,
{
    MultilinearPoly::num_vars(source)
}

fn multilinear_evaluate<F, T>(source: &T, point: &[F]) -> F
where
    F: Field,
    T: MultilinearPoly<F> + ?Sized,
{
    MultilinearPoly::evaluate(source, point)
}

fn multilinear_for_each_row<F, T, V>(source: &T, sigma: usize, mut visit: V)
where
    F: Field,
    T: MultilinearPoly<F> + ?Sized,
    V: for<'row> FnMut(usize, SourceRow<'row, F>),
{
    MultilinearPoly::for_each_row(source, sigma, &mut |row_index, row| {
        visit(row_index, SourceRow::FieldElements(row));
    });
}

fn multilinear_is_one_hot<F, T>(source: &T) -> bool
where
    F: Field,
    T: MultilinearPoly<F> + ?Sized,
{
    MultilinearPoly::is_one_hot(source)
}

fn multilinear_for_each_one<F, T, V>(source: &T, mut visit: V)
where
    F: Field,
    T: MultilinearPoly<F> + ?Sized,
    V: FnMut(usize),
{
    MultilinearPoly::for_each_one(source, &mut visit);
}

fn multilinear_fold_rows<F, T>(source: &T, left: &[F], sigma: usize) -> Vec<F>
where
    F: Field,
    T: MultilinearPoly<F> + ?Sized,
{
    MultilinearPoly::fold_rows(source, left, sigma)
}

macro_rules! impl_commitment_source_for_multilinear {
    ($ty:ty) => {
        impl<F: Field> CommitmentSource<F> for $ty {
            fn num_vars(&self) -> usize {
                multilinear_num_vars(self)
            }

            fn evaluate(&self, point: &[F]) -> F {
                multilinear_evaluate(self, point)
            }

            fn for_each_row<V>(&self, sigma: usize, visit: V)
            where
                V: for<'row> FnMut(usize, SourceRow<'row, F>),
            {
                multilinear_for_each_row(self, sigma, visit);
            }

            fn is_one_hot(&self) -> bool {
                multilinear_is_one_hot(self)
            }

            fn for_each_one<V>(&self, visit: V)
            where
                V: FnMut(usize),
            {
                multilinear_for_each_one(self, visit);
            }

            fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
                multilinear_fold_rows(self, left, sigma)
            }
        }
    };
}

impl_commitment_source_for_multilinear!(Polynomial<F>);
impl_commitment_source_for_multilinear!(Vec<F>);
impl_commitment_source_for_multilinear!([F]);

impl<F, S> CommitmentSource<F> for RlcSource<F, S>
where
    F: Field,
    S: MultilinearPoly<F>,
{
    fn num_vars(&self) -> usize {
        multilinear_num_vars(self)
    }

    fn evaluate(&self, point: &[F]) -> F {
        multilinear_evaluate(self, point)
    }

    fn for_each_row<V>(&self, sigma: usize, visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, F>),
    {
        multilinear_for_each_row(self, sigma, visit);
    }

    fn is_one_hot(&self) -> bool {
        multilinear_is_one_hot(self)
    }

    fn for_each_one<V>(&self, visit: V)
    where
        V: FnMut(usize),
    {
        multilinear_for_each_one(self, visit);
    }

    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
        multilinear_fold_rows(self, left, sigma)
    }
}

impl<F: Field> CommitmentSource<F> for OneHotPolynomial {
    fn num_vars(&self) -> usize {
        multilinear_num_vars::<F, _>(self)
    }

    fn evaluate(&self, point: &[F]) -> F {
        multilinear_evaluate(self, point)
    }

    fn for_each_row<V>(&self, sigma: usize, visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, F>),
    {
        multilinear_for_each_row(self, sigma, visit);
    }

    fn is_one_hot(&self) -> bool {
        multilinear_is_one_hot::<F, _>(self)
    }

    fn for_each_one<V>(&self, visit: V)
    where
        V: FnMut(usize),
    {
        multilinear_for_each_one::<F, _, _>(self, visit);
    }

    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
        multilinear_fold_rows(self, left, sigma)
    }
}

pub(crate) fn materialize_source_evaluations<F, S>(source: &S) -> Vec<F>
where
    F: Field,
    S: CommitmentSource<F> + ?Sized,
{
    fn flush_one_hot<F: Field>(
        evaluations: &mut Vec<F>,
        pending: &mut Option<(usize, Vec<Vec<Option<usize>>>)>,
    ) {
        let Some((domain_size, chunks)) = pending.take() else {
            return;
        };

        let trace_len = chunks.iter().map(Vec::len).sum::<usize>();
        let start = evaluations.len();
        evaluations.resize(start + trace_len * domain_size, F::zero());

        let mut chunk_offset = 0;
        for chunk in chunks {
            for (column, hot_index) in chunk.iter().enumerate() {
                if let Some(hot_index) = hot_index {
                    evaluations[start + hot_index * trace_len + chunk_offset + column] =
                        F::from_u64(1);
                }
            }
            chunk_offset += chunk.len();
        }
    }

    let mut evaluations = Vec::with_capacity(1usize << source.num_vars());
    let mut one_hot_chunks = None;
    source.for_each_row(source.num_vars(), |_, row| match row {
        SourceRow::FieldElements(values) => {
            flush_one_hot(&mut evaluations, &mut one_hot_chunks);
            evaluations.extend_from_slice(values);
        }
        SourceRow::I128(values) => {
            flush_one_hot(&mut evaluations, &mut one_hot_chunks);
            evaluations.extend(values.iter().map(|&value| F::from_i128(value)));
        }
        SourceRow::U64(values) => {
            flush_one_hot(&mut evaluations, &mut one_hot_chunks);
            evaluations.extend(values.iter().map(|&value| F::from_u64(value)));
        }
        SourceRow::OneHot(row) => {
            let domain_size = 1usize << row.log_domain_size;
            let chunk = match row.entries {
                OneHotEntries::OnePerColumn(indices) => {
                    indices.iter().map(|index| Some(index.get())).collect()
                }
                OneHotEntries::MaybeZero(indices) => indices
                    .iter()
                    .map(|index| index.map(OneHotIndex::get))
                    .collect(),
            };

            match &mut one_hot_chunks {
                Some((existing_domain_size, chunks)) => {
                    assert_eq!(
                        *existing_domain_size, domain_size,
                        "one source changed one-hot domain size during materialization",
                    );
                    chunks.push(chunk);
                }
                None => {
                    one_hot_chunks = Some((domain_size, vec![chunk]));
                }
            }
        }
    });
    flush_one_hot(&mut evaluations, &mut one_hot_chunks);
    evaluations
}

/// A batch of committed sources that can share one traversal.
///
/// This is the no-regression traversal hook for current CycleMajor Dory
/// commitment. The default PCS implementation can ignore it and commit sources
/// one at a time through [`source`](Self::source).
pub trait BatchCommitmentSource<F: Field>: Send + Sync {
    type Id: SourceId;

    /// Borrowed single-source adapter for a source in this batch.
    type Source<'a>: CommitmentSource<F> + 'a
    where
        Self: 'a;

    /// All source ids this batch can expose, in natural protocol order.
    fn source_ids(&self) -> &[Self::Id];

    /// Number of multilinear variables in the selected source.
    fn num_vars(&self, id: Self::Id) -> usize;

    /// Returns a single-source view for backends that do not use batch traversal.
    fn source(&self, id: Self::Id) -> Self::Source<'_>;

    /// Maps a row visitor over many sources while sharing source traversal.
    ///
    /// The returned vector is row-major: `output[row_index][id_index]`.
    fn map_rows<R, V>(&self, sigma: usize, ids: &[Self::Id], visit: V) -> Vec<Vec<R>>
    where
        R: Send,
        V: for<'row> Fn(Self::Id, SourceRow<'row, F>) -> R + Send + Sync;
}
