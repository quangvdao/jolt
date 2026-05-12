//! Source abstractions for commitment backends.
//!
//! A source describes committed data and the traversal shapes a backend may
//! exploit. It does not prescribe the backend's commitment algorithm or
//! parallel schedule.

use jolt_field::Field;
use jolt_poly::MultilinearPoly;

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
/// has a required hot coordinate or may be zero.
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

    /// A row whose entries are one-hot vectors over a small domain.
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

impl<F, T> CommitmentSource<F> for T
where
    F: Field,
    T: MultilinearPoly<F> + ?Sized,
{
    fn num_vars(&self) -> usize {
        MultilinearPoly::num_vars(self)
    }

    fn evaluate(&self, point: &[F]) -> F {
        MultilinearPoly::evaluate(self, point)
    }

    fn for_each_row<V>(&self, sigma: usize, mut visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, F>),
    {
        MultilinearPoly::for_each_row(self, sigma, &mut |row_index, row| {
            visit(row_index, SourceRow::FieldElements(row));
        });
    }

    fn is_one_hot(&self) -> bool {
        MultilinearPoly::is_one_hot(self)
    }

    fn for_each_one<V>(&self, mut visit: V)
    where
        V: FnMut(usize),
    {
        MultilinearPoly::for_each_one(self, &mut visit);
    }

    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
        MultilinearPoly::fold_rows(self, left, sigma)
    }
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
