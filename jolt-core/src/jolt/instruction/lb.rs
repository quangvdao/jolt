use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, SubtableIndices};
use crate::field::JoltField;
use crate::jolt::subtable::identity::IdentitySubtable;
use crate::jolt::subtable::{
    sign_extend::SignExtendSubtable, truncate_overflow::TruncateOverflowSubtable, LassoSubtable,
};
use crate::utils::instruction_utils::chunk_operand_usize;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct LBInstruction<const WORD_SIZE: usize>(pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for LBInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (0, self.0)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        assert!(M >= 1 << 8);

        let byte = vals[0];
        let sign_extension = vals[1];

        let mut result = byte;
        for i in 1..C {
            result += F::from_u64(1 << (8 * i)).unwrap() * sign_extension;
        }
        result
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        // This assertion ensures that we only need one TruncateOverflowSubtable
        assert!(M >= 1 << 8);
        vec![
            (
                // Truncate all but the lowest eight bits of the last chunk,
                // which contains the lower 8 bits of the loaded value.
                Box::new(TruncateOverflowSubtable::<F, 8>::new()),
                SubtableIndices::from(C - 1),
            ),
            (
                // Sign extend the lowest eight bits of the last chunk,
                // which contains the lower 8 bits of the loaded value.
                Box::new(SignExtendSubtable::<F, 8>::new()),
                SubtableIndices::from(C - 1),
            ),
            (
                // Not used for lookup, but this implicitly range-checks
                // the remaining query chunks
                Box::new(IdentitySubtable::<F>::new()),
                SubtableIndices::from(0..C - 1),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_operand_usize(self.0, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        // Sign-extend lower 8 bits of the loaded value
        // This is the same for both 32-bit and 64-bit word sizes
        (self.0 & 0xff) as i8 as i32 as u32 as u64
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        if WORD_SIZE == 32 {
            Self(rng.next_u32() as u64)
        } else if WORD_SIZE == 64 {
            Self(rng.next_u64())
        } else {
            panic!("Only 32-bit and 64-bit word sizes are supported");
        }
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use super::LBInstruction;
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    #[test]
    fn lb_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let instruction = LBInstruction::<WORD_SIZE>(x);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            LBInstruction::<WORD_SIZE>(0),
            LBInstruction::<WORD_SIZE>(1),
            LBInstruction::<WORD_SIZE>(100),
            LBInstruction::<WORD_SIZE>(u32_max),
            LBInstruction::<WORD_SIZE>(1 << 8),
            LBInstruction::<WORD_SIZE>(u32_max - 100),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    // Doesn't work for now
    // #[test]
    // fn lb_instruction_64_e2e() {
    //     let mut rng = test_rng();
    //     const C: usize = 8;
    //     const M: usize = 1 << 16;
    //     const WORD_SIZE: usize = 64;

    //     // Random
    //     for _ in 0..256 {
    //         let x = rng.next_u64();
    //         let instruction = LBInstruction::<WORD_SIZE>(x);
    //         jolt_instruction_test!(instruction);
    //     }

    //     // Edge cases
    //     let u64_max: u64 = u64::MAX;
    //     let instructions = vec![
    //         LBInstruction::<WORD_SIZE>(0),
    //         LBInstruction::<WORD_SIZE>(1),
    //         LBInstruction::<WORD_SIZE>(100),
    //         LBInstruction::<WORD_SIZE>(u64_max),
    //         LBInstruction::<WORD_SIZE>(1 << 8),
    //         LBInstruction::<WORD_SIZE>(1 << 32 - 1),
    //     ];
    //     for instruction in instructions {
    //         jolt_instruction_test!(instruction);
    //     }
    // }
}
