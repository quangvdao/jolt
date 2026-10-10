//! Structured equations `Az · Bz = Cz`, with the sparse matrices derived from
//! the same lane and polynomial forms.

use crate::layout::Layout;
use crate::words::{column, Lane, WitnessRow, WITNESS_COLUMNS};
use jolt_field::F128;
use jolt_r1cs::{ConstraintMatrices, SparseRow};
use thiserror::Error;

/// The equation family, in fixed row order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowGroup {
    Adder,
    And,
    LessThan,
    KeyBits,
    KeysAgreeAbove,
    KeysEqual,
    RdResidual,
    RamResidual,
    NextPCResidual,
    ControlResidual,
    OneHot,
}

/// Sixty-four rows whose A and B columns are masked bit lanes and whose C
/// columns are the unmasked lane: `(z[a] & mask) · (z[b] & mask) = z[c]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneRows {
    pub a: Lane,
    pub b: Lane,
    pub c: Lane,
    pub ab_mask: u64,
    pub group: RowGroup,
}

impl LaneRows {
    /// Returns `Az`, `Bz`, `Cz`, with bit i the value of row i in the family.
    #[inline]
    pub fn values(&self, z: &WitnessRow) -> [u64; 3] {
        [
            z.lane(self.a) & self.ab_mask,
            z.lane(self.b) & self.ab_mask,
            z.lane(self.c),
        ]
    }
}

/// Invalid domain of `Σ (x^(stride·t + shift) + [complement]) z[start+t]`.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PackedTermError {
    #[error("packed term length {len} is outside 1..=64")]
    LengthOutOfRange { len: u8 },
    #[error("packed term columns {start}..{start}+{len} leave the 1024-column witness")]
    ColumnsOutOfRange { start: u16, len: u8 },
    #[error("packed term exponent for length {len}, stride {stride}, shift {shift} reaches 128")]
    ExponentOutOfRange { len: u8, stride: u8, shift: u8 },
}

/// `Σ_{t<len} (x^(stride·t + shift) + [complement]) · z[start+t]`.
/// The checked domain has 1–64 columns within the witness and exponents <128.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedTerm {
    start: u16,
    len: u8,
    stride: u8,
    shift: u8,
    complement: bool,
}

impl PackedTerm {
    /// Checks the column interval and the largest polynomial exponent.
    pub const fn new(
        start: u16,
        len: u8,
        stride: u8,
        shift: u8,
        complement: bool,
    ) -> Result<Self, PackedTermError> {
        if len == 0 || len > 64 {
            return Err(PackedTermError::LengthOutOfRange { len });
        }
        if start as usize + len as usize > WITNESS_COLUMNS {
            return Err(PackedTermError::ColumnsOutOfRange { start, len });
        }
        if stride as u16 * (len - 1) as u16 + shift as u16 >= 128 {
            return Err(PackedTermError::ExponentOutOfRange { len, stride, shift });
        }
        Ok(Self::table(start, len, stride, shift, complement))
    }

    const fn table(start: u16, len: u8, stride: u8, shift: u8, complement: bool) -> Self {
        Self {
            start,
            len,
            stride,
            shift,
            complement,
        }
    }

    /// First input column.
    #[inline]
    pub const fn start(self) -> u16 {
        self.start
    }
    /// Number of consecutive input columns, in 1..=64.
    #[inline]
    pub const fn length(self) -> u8 {
        self.len
    }
    /// Distance between successive polynomial exponents.
    #[inline]
    pub const fn stride(self) -> u8 {
        self.stride
    }
    /// Exponent of the first input column.
    #[inline]
    pub const fn shift(self) -> u8 {
        self.shift
    }
    /// Whether each coefficient additionally contains its constant monomial.
    #[inline]
    pub const fn complement(self) -> bool {
        self.complement
    }

    #[inline]
    fn value(self, z: &WitnessRow) -> u128 {
        let start = usize::from(self.start);
        let offset = start % 64;
        let low = z.0.get(start / 64).copied().unwrap_or(0) >> offset;
        let high = if offset != 0 {
            z.0.get(start / 64 + 1).copied().unwrap_or(0) << (64 - offset)
        } else {
            0
        };
        let mask = u64::MAX >> (64 - self.len);
        let v = (low | high) & mask;
        let mut spread = match self.stride {
            0 => u128::from(v.count_ones() & 1),
            1 => u128::from(v),
            _ => {
                let mut result = 0;
                let mut remaining = v;
                while remaining != 0 {
                    let t = remaining.trailing_zeros();
                    result ^= 1u128 << (u32::from(self.stride) * t);
                    remaining &= remaining - 1;
                }
                result
            }
        } << self.shift;
        if self.complement {
            spread ^= u128::from(v.count_ones() & 1);
        }
        spread
    }

    fn append_coefficients(self, out: &mut SparseRow<F128>) {
        for t in 0..self.len {
            let exponent = u32::from(self.stride) * u32::from(t) + u32::from(self.shift);
            let raw = (1u128 << exponent) ^ u128::from(self.complement);
            out.push((
                usize::from(self.start) + usize::from(t),
                F128::from_raw(raw),
            ));
        }
    }
}

/// XOR of packed terms, plus the witness column `ONE` when `one` is set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PackedForm {
    pub one: bool,
    pub terms: Vec<PackedTerm>,
}

impl PackedForm {
    /// Evaluates the polynomial coefficients using shifts and XORs only.
    /// Constants read `z[ONE]` even on a noncanonical witness.
    #[inline]
    pub fn values(&self, z: &WitnessRow) -> F128 {
        let constant = self.one && z.bit(column::ONE).unwrap_or(false);
        let raw = self
            .terms
            .iter()
            .fold(u128::from(constant), |sum, term| sum ^ term.value(z));
        F128::from_raw(raw)
    }

    fn coefficients(&self) -> SparseRow<F128> {
        let mut out = Vec::new();
        if self.one {
            out.push((column::ONE, F128::from_raw(1)));
        }
        for &term in &self.terms {
            term.append_coefficients(&mut out);
        }
        out
    }
}

/// One equation over `F128`, `a(z) · b(z) = c(z)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedRow {
    pub a: PackedForm,
    pub b: PackedForm,
    pub c: PackedForm,
    pub group: RowGroup,
}

impl PackedRow {
    /// Returns `Az`, `Bz`, `Cz`; this evaluation performs no field multiplication.
    #[inline]
    pub fn values(&self, z: &WitnessRow) -> [F128; 3] {
        [self.a.values(z), self.b.values(z), self.c.values(z)]
    }
}

/// Stack bitset of failing row indices, in ascending iteration order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowSet {
    bits: [u64; 3],
}

impl RowSet {
    /// Whether every checked equation holds.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|word| *word == 0)
    }
    /// Whether the given equation failed; out-of-range indices return false.
    #[inline]
    pub fn contains(&self, row: usize) -> bool {
        self.bits
            .get(row / 64)
            .is_some_and(|word| word & (1u64 << (row % 64)) != 0)
    }
    /// Failed row indices in increasing order.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.bits.iter().enumerate().flat_map(|(block, &word)| {
            (0..64).filter_map(move |bit| (word & (1u64 << bit) != 0).then_some(block * 64 + bit))
        })
    }
    fn insert(&mut self, row: usize) {
        if let Some(word) = self.bits.get_mut(row / 64) {
            *word |= 1u64 << (row % 64);
        }
    }
}

/// The first equation that failed, with its family.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("constraint row {row} in group {group:?} failed")]
pub struct RowFailure {
    pub row: usize,
    pub group: RowGroup,
}

/// Rows 0–127 are the Adder and And lanes; rows 128–135 check comparison and
/// residuals; rows 136+c check chunk c's one-hot polynomial equation.
///
/// | Rows | Equation |
/// |---|---|
/// | 0..63 | CarryLeft[i] AND CarryRight[i] = CarryStep[i] |
/// | 63 | 0 = CarryStep[63] |
/// | 64..128 | AndLeft[i] AND AndRight[i] = AndOut[i] |
/// | 128 | KeysDiffer AND RightKeyBit = LessThan |
/// | 129 | (LeftKeyBit XOR RightKeyBit XOR KeysDiffer) · ONE = 0 |
/// | 130 | KeysDiffer · pack(KeyDiffAbove) = 0 |
/// | 131 | (ONE XOR KeysDiffer) · pack(KeyDiff) = 0 |
/// | 132, 133, 134 | pack(RdResidual), pack(RamResidual), pack(NextPCResidual) · ONE = 0 |
/// | 135 | pack(ControlResidual) · ONE = 0 |
/// | 136+c | h_1(c) · h_2(c) = h_3(c) |
///
/// Here `pack(W) = F128::from_raw(W as u128)` and
/// `h_m(c) = ONE + Σ_{k=1..n} (1 + x^(m·k)) Ra_c[k]`, with chunks in
/// `Layout::chunks()` order. All additions in these equations are XOR.
///
/// On a canonical witness these are local equations. The surrounding protocol
/// must bind `BytecodeRa` to the supplied row, validate `NextPC` as a bytecode
/// address, enforce `RdWriteValue = old_rd XOR ((1 XOR Store) AND Inc)`, and
/// enforce `ram_post = ram_pre XOR (Store AND Inc)`.
#[derive(Clone, Debug)]
pub struct RowSystem {
    lanes: [LaneRows; 2],
    packed: Vec<PackedRow>,
}

impl RowSystem {
    /// Rows 0–129 have coefficients and evaluations in `F_2`.
    pub const F2_ROWS: usize = 130;

    /// Constructs `136 + ceil(log_K_bytecode/4) + ceil(log_K_ram/4) + 2`
    /// equations; stored digits k use coefficients `1 + x^(m·k)`, m=1,2,3.
    pub fn new(layout: &Layout) -> Self {
        let bit = |col: usize| PackedForm {
            one: false,
            terms: vec![PackedTerm::table(col as u16, 1, 0, 0, false)],
        };
        let block = |lane: Lane| PackedForm {
            one: false,
            terms: vec![PackedTerm::table(lane as u16 * 64, 64, 1, 0, false)],
        };
        let one = PackedForm {
            one: true,
            terms: vec![],
        };
        let empty = PackedForm::default();
        let keys_differ = bit(column::BITS + layout.keys_differ());
        let mut not_keys_differ = keys_differ.clone();
        not_keys_differ.one = true;
        let mut key_bits = keys_differ.clone();
        key_bits.terms.push(PackedTerm::table(
            column::LEFT_KEY_BIT as u16,
            2,
            0,
            0,
            false,
        ));
        let mut packed = vec![
            PackedRow {
                a: keys_differ.clone(),
                b: bit(column::RIGHT_KEY_BIT),
                c: bit(column::LESS_THAN),
                group: RowGroup::LessThan,
            },
            PackedRow {
                a: key_bits,
                b: one.clone(),
                c: empty.clone(),
                group: RowGroup::KeyBits,
            },
            PackedRow {
                a: keys_differ,
                b: block(Lane::KeyDiffAbove),
                c: empty.clone(),
                group: RowGroup::KeysAgreeAbove,
            },
            PackedRow {
                a: not_keys_differ,
                b: block(Lane::KeyDiff),
                c: empty.clone(),
                group: RowGroup::KeysEqual,
            },
        ];
        for (lane, group) in [
            (Lane::RdResidual, RowGroup::RdResidual),
            (Lane::RamResidual, RowGroup::RamResidual),
            (Lane::NextPCResidual, RowGroup::NextPCResidual),
        ] {
            packed.push(PackedRow {
                a: block(lane),
                b: one.clone(),
                c: empty.clone(),
                group,
            });
        }
        packed.push(PackedRow {
            a: PackedForm {
                one: false,
                terms: vec![PackedTerm::table(
                    column::CONTROL_RESIDUAL as u16,
                    11,
                    1,
                    0,
                    false,
                )],
            },
            b: one,
            c: empty,
            group: RowGroup::ControlResidual,
        });
        for chunk in layout.chunks() {
            let form = |m| PackedForm {
                one: true,
                terms: vec![PackedTerm::table(
                    column::BITS as u16 + chunk.start(),
                    chunk.indicators() as u8,
                    m,
                    m,
                    true,
                )],
            };
            packed.push(PackedRow {
                a: form(1),
                b: form(2),
                c: form(3),
                group: RowGroup::OneHot,
            });
        }
        Self {
            lanes: [
                LaneRows {
                    a: Lane::CarryLeft,
                    b: Lane::CarryRight,
                    c: Lane::CarryStep,
                    ab_mask: u64::MAX >> 1,
                    group: RowGroup::Adder,
                },
                LaneRows {
                    a: Lane::AndLeft,
                    b: Lane::AndRight,
                    c: Lane::AndOut,
                    ab_mask: u64::MAX,
                    group: RowGroup::And,
                },
            ],
            packed,
        }
    }

    /// The two 64-row lane families, starting at rows 0 and 64.
    pub fn lane_rows(&self) -> &[LaneRows; 2] {
        &self.lanes
    }
    /// Packed equations, starting at row 128.
    pub fn packed_rows(&self) -> &[PackedRow] {
        &self.packed
    }
    /// Total equations, including one per address and position chunk.
    pub fn num_rows(&self) -> usize {
        128 + self.packed.len()
    }
    /// Family of a row, or `None` beyond the equation list.
    pub fn group_of(&self, row: usize) -> Option<RowGroup> {
        if row < 128 {
            self.lanes.get(row / 64).map(|family| family.group)
        } else {
            self.packed.get(row - 128).map(|eq| eq.group)
        }
    }

    /// Checks the listed equations on a canonical witness. This does not bind
    /// `BytecodeRa` to the row, validate `NextPC` as a bytecode address, or enforce
    /// the register and RAM update identities documented on [`RowSystem`].
    pub fn failing_rows(&self, z: &WitnessRow) -> RowSet {
        let mut failures = RowSet::default();
        for (family, lane) in self.lanes.iter().enumerate() {
            let [a, b, c] = lane.values(z);
            let mut residual = (a & b) ^ c;
            while residual != 0 {
                failures.insert(family * 64 + residual.trailing_zeros() as usize);
                residual &= residual - 1;
            }
        }
        for (r, eq) in self.packed.iter().enumerate() {
            let [a, b, c] = eq.values(z);
            if a * b != c {
                failures.insert(128 + r);
            }
        }
        failures
    }

    /// Returns the lowest failed equation. The local contract and surrounding
    /// protocol obligations are those of [`RowSystem::failing_rows`].
    pub fn check(&self, z: &WitnessRow) -> Result<(), RowFailure> {
        for row in self.failing_rows(z).iter() {
            if let Some(group) = self.group_of(row) {
                return Err(RowFailure { row, group });
            }
        }
        Ok(())
    }

    /// Expands these forms into A, B, C sparse matrices with 1024 columns.
    /// A constant coefficient references column `ONE` literally.
    pub fn to_matrices(&self) -> ConstraintMatrices<F128> {
        let mut a = Vec::with_capacity(self.num_rows());
        let mut b = Vec::with_capacity(self.num_rows());
        let mut c = Vec::with_capacity(self.num_rows());
        for lane in &self.lanes {
            for i in 0..64 {
                let ab = |which: Lane| {
                    if lane.ab_mask & (1u64 << i) != 0 {
                        vec![(which as usize * 64 + i, F128::from_raw(1))]
                    } else {
                        vec![]
                    }
                };
                a.push(ab(lane.a));
                b.push(ab(lane.b));
                c.push(vec![(lane.c as usize * 64 + i, F128::from_raw(1))]);
            }
        }
        for row in &self.packed {
            a.push(row.a.coefficients());
            b.push(row.b.coefficients());
            c.push(row.c.coefficients());
        }
        ConstraintMatrices {
            num_constraints: self.num_rows(),
            num_vars: WITNESS_COLUMNS,
            a,
            b,
            c,
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "test assertions use checked fixtures and panic on failure"
)]
mod tests {
    use super::{PackedForm, PackedTerm, PackedTermError, RowGroup, RowSystem};
    use crate::layout::Layout;
    use crate::words::{column, WitnessRow, WITNESS_COLUMNS};
    use jolt_field::F128;
    use jolt_r1cs::{ConstraintMatrices, SparseRow};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn packed_descriptor_domains_and_table_entries() {
        assert_eq!(
            PackedTerm::new(0, 0, 1, 0, false),
            Err(PackedTermError::LengthOutOfRange { len: 0 })
        );
        assert_eq!(
            PackedTerm::new(0, 65, 1, 0, false),
            Err(PackedTermError::LengthOutOfRange { len: 65 })
        );
        assert_eq!(
            PackedTerm::new(1024, 1, 0, 0, false),
            Err(PackedTermError::ColumnsOutOfRange {
                start: 1024,
                len: 1
            })
        );
        assert_eq!(
            PackedTerm::new(1023, 2, 0, 0, false),
            Err(PackedTermError::ColumnsOutOfRange {
                start: 1023,
                len: 2
            })
        );
        assert_eq!(
            PackedTerm::new(0, 64, 2, 2, false),
            Err(PackedTermError::ExponentOutOfRange {
                len: 64,
                stride: 2,
                shift: 2
            })
        );
        assert_eq!(
            PackedTerm::new(0, 1, 0, 128, false),
            Err(PackedTermError::ExponentOutOfRange {
                len: 1,
                stride: 0,
                shift: 128
            })
        );
        assert!(PackedTerm::new(1023, 1, 255, 127, true).is_ok());
        assert!(PackedTerm::new(0, 64, 2, 1, true).is_ok());
        for log_bytecode in 1..=32 {
            for log_ram in 1..=61 {
                if let Ok(layout) = Layout::new(log_bytecode, log_ram, 0) {
                    let rows = RowSystem::new(&layout);
                    for row in rows.packed_rows() {
                        for form in [&row.a, &row.b, &row.c] {
                            for term in &form.terms {
                                assert_eq!(
                                    PackedTerm::new(
                                        term.start(),
                                        term.length(),
                                        term.stride(),
                                        term.shift(),
                                        term.complement()
                                    ),
                                    Ok(*term)
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn row_counts_order_and_failure_set() {
        for (log_bytecode, log_ram, expected) in
            [(20, 20, 148), (20, 23, 149), (21, 21, 150), (22, 24, 150)]
        {
            let layout = Layout::new(log_bytecode, log_ram, 0).unwrap();
            let system = RowSystem::new(&layout);
            assert_eq!(system.num_rows(), expected);
            assert_eq!(system.group_of(0), Some(RowGroup::Adder));
            assert_eq!(system.group_of(63), Some(RowGroup::Adder));
            assert_eq!(system.group_of(64), Some(RowGroup::And));
            assert_eq!(system.group_of(127), Some(RowGroup::And));
            for (row, group) in [
                RowGroup::LessThan,
                RowGroup::KeyBits,
                RowGroup::KeysAgreeAbove,
                RowGroup::KeysEqual,
                RowGroup::RdResidual,
                RowGroup::RamResidual,
                RowGroup::NextPCResidual,
                RowGroup::ControlResidual,
            ]
            .into_iter()
            .enumerate()
            {
                assert_eq!(system.group_of(128 + row), Some(group));
            }
            assert_eq!(system.group_of(expected - 1), Some(RowGroup::OneHot));
            assert_eq!(system.group_of(expected), None);
            assert_eq!(system.group_of(usize::MAX), None);
            let mut z = WitnessRow([0; 16]);
            z.0[0] = 1;
            assert!(system.check(&z).is_ok());
            z.0[3] = 1 | (1u64 << 63);
            let failures = system.failing_rows(&z);
            assert_eq!(failures.iter().collect::<Vec<_>>(), [0, 63]);
            assert!(failures.contains(0));
            assert!(!failures.contains(usize::MAX));
            assert_eq!(system.check(&z).unwrap_err().row, 0);
        }
        assert_eq!(RowSystem::F2_ROWS, 130);
    }

    #[test]
    fn one_hot_equation_matches_weight_for_every_pattern() {
        for bits in 1..=4 {
            let layout = Layout::new(bits, 1, 0).unwrap();
            let system = RowSystem::new(&layout);
            let chunk = layout.bytecode_ra()[0];
            let eq = &system.packed_rows()[8];
            assert_eq!(eq.group, RowGroup::OneHot);
            let n = chunk.indicators();
            for pattern in 0u64..(1u64 << n) {
                let mut committed = [0u64; 4];
                for i in 0..n {
                    let col = usize::from(chunk.start()) + i;
                    committed[col / 64] |= ((pattern >> i) & 1) << (col % 64);
                }
                let mut z = WitnessRow([0; 16]);
                z.0[0] = 1;
                z.0[12..].copy_from_slice(&committed);
                let [a, b, c] = eq.values(&z);
                assert_eq!(
                    a * b == c,
                    chunk.full(&committed).is_power_of_two(),
                    "width={bits}, pattern={pattern}"
                );
            }
        }
    }

    fn assert_matrix_values(
        system: &RowSystem,
        matrices: &ConstraintMatrices<F128>,
        z: &WitnessRow,
        dense: &mut [F128],
    ) {
        for (col, value) in dense.iter_mut().enumerate() {
            *value = F128::from_raw(u128::from(z.bit(col).unwrap()));
        }
        let dot = |row: &SparseRow<F128>| {
            row.iter()
                .fold(F128::from_raw(0), |sum, &(col, coefficient)| {
                    sum + coefficient * dense[col]
                })
        };
        for (family, lane) in system.lane_rows().iter().enumerate() {
            let [a, b, c] = lane.values(z);
            for i in 0..64 {
                let r = family * 64 + i;
                assert_eq!(
                    [
                        dot(&matrices.a[r]),
                        dot(&matrices.b[r]),
                        dot(&matrices.c[r])
                    ],
                    [
                        F128::from_raw(u128::from((a >> i) & 1)),
                        F128::from_raw(u128::from((b >> i) & 1)),
                        F128::from_raw(u128::from((c >> i) & 1))
                    ]
                );
            }
        }
        for (offset, row) in system.packed_rows().iter().enumerate() {
            let r = 128 + offset;
            assert_eq!(
                row.values(z),
                [
                    dot(&matrices.a[r]),
                    dot(&matrices.b[r]),
                    dot(&matrices.c[r])
                ],
                "row {r}"
            );
        }
    }

    #[test]
    fn structured_values_equal_sparse_matrices_on_arbitrary_witnesses() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x9fa2_3c19);
        for (log_bytecode, log_ram) in [(20, 20), (21, 23)] {
            let layout = Layout::new(log_bytecode, log_ram, 0).unwrap();
            let system = RowSystem::new(&layout);
            let matrices = system.to_matrices();
            assert!(matrices.validate().is_ok());
            let mut dense = vec![F128::from_raw(0); WITNESS_COLUMNS];
            for _ in 0..10_000 {
                let mut z = WitnessRow(rng.gen());
                z.0[0] |= 1;
                assert_matrix_values(&system, &matrices, &z, &mut dense);
            }
            let mut z = WitnessRow(rng.gen());
            z.0[0] &= !1;
            assert!(!z.bit(column::ONE).unwrap());
            assert_matrix_values(&system, &matrices, &z, &mut dense);
        }
    }

    #[test]
    fn packed_form_crosses_lanes_and_reaches_exponent_127() {
        let mut z = WitnessRow([0; 16]);
        z.0[0] = (1u64 << 63) | 1;
        z.0[1] = 1 | (1u64 << 62);
        let form = PackedForm {
            one: true,
            terms: vec![PackedTerm::new(63, 64, 2, 1, true).unwrap()],
        };
        let expected = 1 ^ ((1u128 << 1) ^ 1) ^ ((1u128 << 3) ^ 1) ^ ((1u128 << 127) ^ 1);
        assert_eq!(form.values(&z).to_raw(), expected);
        let parity = PackedForm {
            one: false,
            terms: vec![PackedTerm::new(63, 64, 0, 127, false).unwrap()],
        };
        assert_eq!(parity.values(&z).to_raw(), 1u128 << 127);
    }
}
