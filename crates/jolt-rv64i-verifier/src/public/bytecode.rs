//! Public bytecode folds use split equality tables and one pass over valid rows.

use jolt_field::JoltField;
use jolt_poly::EqPolynomial;
use jolt_rv64i_arith::{Bytecode, BytecodeColumn, BytecodeRow};

use crate::claims::bytecode_read::{BytecodeReadAddressChallenges, BytecodeReadAddressInputClaims};
use crate::points::{self, PointsError};

/// Earlier points needed by both phases. `new` validates their dimensions and
/// derives the overlapping kind points from the short point of batch 3a.
#[derive(Clone)]
pub struct BytecodeReadPoints<F> {
    pub r_bit: Vec<F>,
    pub q_variant: Vec<F>,
    pub q_shift: Vec<F>,
    pub q_access: Vec<F>,
    pub q_key: Vec<F>,
    pub a_reg: Vec<F>,
    pub r_3: Vec<F>,
    pub r_4: Vec<F>,
    pub r_5: Vec<F>,
}

impl<F: JoltField> BytecodeReadPoints<F> {
    pub fn new(
        x: &[F],
        a_reg: Vec<F>,
        r_3: Vec<F>,
        r_4: Vec<F>,
        r_5: Vec<F>,
    ) -> Result<Self, PointsError> {
        if x.len() != 17 {
            return Err(PointsError::Dimension {
                expected: 17,
                actual: x.len(),
            });
        }
        let slice = |start, end| {
            x.get(start..end)
                .map(<[F]>::to_vec)
                .ok_or(PointsError::Dimension {
                    expected: end,
                    actual: x.len(),
                })
        };
        let points = Self {
            r_bit: slice(0, 6)?,
            q_variant: slice(11, 17)?,
            q_shift: slice(12, 15)?,
            q_access: slice(13, 17)?,
            q_key: slice(14, 17)?,
            a_reg,
            r_3,
            r_4,
            r_5,
        };
        points.validate()?;
        Ok(points)
    }

    pub fn validate(&self) -> Result<(), PointsError> {
        for (point, expected) in [
            (&self.r_bit, 6),
            (&self.q_variant, 6),
            (&self.q_shift, 3),
            (&self.q_access, 4),
            (&self.q_key, 3),
            (&self.a_reg, 5),
            (&self.r_4, self.r_3.len()),
            (&self.r_5, self.r_3.len()),
        ] {
            if point.len() != expected {
                return Err(PointsError::Dimension {
                    expected,
                    actual: point.len(),
                });
            }
        }
        Ok(())
    }

    pub fn input_points(&self) -> BytecodeReadAddressInputClaims<Vec<F>> {
        let word = || self.r_bit.iter().chain(&self.r_3).copied().collect();
        let router = |point: &[F]| point.iter().chain(&self.r_3).copied().collect();
        let read = || self.a_reg.iter().chain(&self.r_4).copied().collect();
        BytecodeReadAddressInputClaims {
            imm: word(),
            fall_through_pc: word(),
            pc_plus_imm: word(),
            pc: word(),
            next_pc: word(),
            variant: router(&self.q_variant),
            shift_kind: router(&self.q_shift),
            access_kind: router(&self.q_access),
            key_kind: router(&self.q_key),
            branch: self.r_3.clone(),
            rs1_ra: read(),
            rs2_ra: read(),
            rd_wa_read: read(),
            rd_wa_write: self.a_reg.iter().chain(&self.r_5).copied().collect(),
            store: self.r_5.clone(),
        }
    }
}

/// Prepared public weights. Kind and register coefficients are folded into
/// their small equality tables before a row is read.
pub struct BytecodeWeights<F> {
    lift: Vec<F>,
    variant: Vec<F>,
    shift: Vec<F>,
    access: Vec<F>,
    key: Vec<F>,
    rs1: Vec<F>,
    rs2: Vec<F>,
    rd_read: Vec<F>,
    rd_write: Vec<F>,
    coefficients: BytecodeReadAddressChallenges<F>,
}

impl<F: JoltField> BytecodeWeights<F> {
    pub fn new(
        points: &BytecodeReadPoints<F>,
        coefficients: &BytecodeReadAddressChallenges<F>,
    ) -> Result<Self, PointsError> {
        points.validate()?;
        let weighted = |point: &[F], coefficient| {
            Ok::<_, PointsError>(
                points::eq_table(point)?
                    .into_iter()
                    .map(|weight| coefficient * weight)
                    .collect(),
            )
        };
        let register = points::eq_table(&points.a_reg)?;
        let registers = |coefficient| {
            register
                .iter()
                .map(|weight| coefficient * *weight)
                .collect()
        };
        Ok(Self {
            lift: points::eq_table(&points.r_bit)?,
            variant: weighted(&points.q_variant, coefficients.variant)?,
            shift: weighted(&points.q_shift, coefficients.shift_kind)?,
            access: weighted(&points.q_access, coefficients.access_kind)?,
            key: weighted(&points.q_key, coefficients.key_kind)?,
            rs1: registers(coefficients.rs1_ra),
            rs2: registers(coefficients.rs2_ra),
            rd_read: registers(coefficients.rd_wa_read),
            rd_write: registers(coefficients.rd_wa_write),
            coefficients: coefficients.clone(),
        })
    }

    pub fn lift(&self, word: u64) -> F {
        self.lift
            .iter()
            .enumerate()
            .filter(|(bit, _)| word & (1_u64 << bit) != 0)
            .map(|(_, weight)| *weight)
            .sum()
    }

    fn selector(table: &[F], index: usize) -> Result<F, PointsError> {
        table.get(index).copied().ok_or(PointsError::Index {
            index,
            variables: 6,
        })
    }

    fn row_values(&self, row: &BytecodeRow) -> Result<[F; 4], PointsError> {
        let Some(variant) = row.variant else {
            return Ok([F::zero(); 4]);
        };
        let pc = self.lift(row.column(BytecodeColumn::PC));
        let mut router = self.coefficients.imm * self.lift(row.column(BytecodeColumn::Imm))
            + self.coefficients.fall_through_pc
                * self.lift(row.column(BytecodeColumn::FallThroughPC))
            + self.coefficients.pc_plus_imm * self.lift(row.column(BytecodeColumn::PCPlusImm))
            + self.coefficients.pc * pc
            + Self::selector(&self.variant, variant.index())?;
        if let Some(shift) = variant.shift() {
            router += Self::selector(&self.shift, shift.kind.index())?;
        }
        if let Some(access) = variant.access().and_then(|access| access.kind) {
            router += Self::selector(&self.access, access.index())?;
        }
        if let Some(key) = variant.key_kind() {
            router += Self::selector(&self.key, key.index())?;
        }
        if variant.branch().is_some() {
            router += self.coefficients.branch;
        }
        let read = Self::selector(&self.rs1, usize::from(row.rs1))?
            + Self::selector(&self.rs2, usize::from(row.rs2))?
            + Self::selector(&self.rd_read, usize::from(row.rd))?;
        let val = Self::selector(&self.rd_write, usize::from(row.rd))?
            + if variant.is_store() {
                self.coefficients.store
            } else {
                F::zero()
            };
        Ok([router, read, val, pc])
    }

    fn fold_pc(&self, [router, read, val, pc]: [F; 4]) -> [F; 5] {
        [
            router,
            read,
            val,
            self.coefficients.entry * pc,
            self.coefficients.next * pc,
        ]
    }

    /// Five `H_t[k]` values in `CycleWeight` order; invalid rows give zero.
    pub fn row(&self, row: &BytecodeRow) -> Result<[F; 5], PointsError> {
        self.row_values(row).map(|values| self.fold_pc(values))
    }

    /// Evaluates all five folds at the bytecode point. The split tables use
    /// at most `2^(ceil(b/2)+1)` entries. A valid row costs nine multiplications:
    /// four word coefficients, one split weight, and four accumulations. Entry
    /// and Next share the PC accumulation and cost two final multiplications.
    pub fn evaluate(&self, bytecode: &Bytecode, a_bc: &[F]) -> Result<[F; 5], PointsError> {
        if a_bc.len() != bytecode.log_K() || a_bc.len() > 24 {
            return Err(PointsError::Dimension {
                expected: bytecode.log_K().min(24),
                actual: a_bc.len(),
            });
        }
        let split = a_bc.len() / 2;
        let low = a_bc.get(..split).ok_or(PointsError::Dimension {
            expected: split,
            actual: a_bc.len(),
        })?;
        let high = a_bc.get(split..).ok_or(PointsError::Dimension {
            expected: split,
            actual: a_bc.len(),
        })?;
        let low_table = EqPolynomial::new(points::to_high_to_low(low)).evaluations();
        let high_table = EqPolynomial::new(points::to_high_to_low(high)).evaluations();
        let mask = (1_usize << split) - 1;
        let mut sums = [F::zero(); 4];
        for (index, row) in bytecode
            .rows()
            .iter()
            .enumerate()
            .filter(|(_, row)| row.variant.is_some())
        {
            let weight = Self::selector(&low_table, index & mask)?
                * Self::selector(&high_table, index >> split)?;
            for (sum, value) in sums.iter_mut().zip(self.row_values(row)?) {
                *sum += weight * value;
            }
        }
        Ok(self.fold_pc(sums))
    }
}
