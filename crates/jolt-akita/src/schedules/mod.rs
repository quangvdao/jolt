//! Jolt-owned external schedule catalog generation.
//!
//! Checked-in `.aks` artifacts are ordinary runtime data, not Rust modules.
//! Regenerate them with:
//!
//! ```text
//! cargo run --release -p jolt-akita --bin gen_jolt_schedules -- crates/jolt-akita/schedules
//! # Only the full-field dense family:
//! cargo run --release -p jolt-akita --bin gen_jolt_schedules -- crates/jolt-akita/schedules dense-full
//! ```

/// Emit-spec construction shared by the generator and drift tests.
pub mod emit {
    use std::path::PathBuf;

    use akita_config::{policy_of, CommitmentConfig};
    use akita_params::{
        FoldSchedule, OpeningClaimsLayout, PolynomialGroupLayout, ScheduleLookupKey,
    };
    use akita_pcs::AkitaError;
    use akita_planner::emit::GroupedGenerationRequest;
    use akita_planner::EmitSpec;

    use crate::configs::{
        AkitaOneHotChunkProfile, JoltDenseBounded, JoltDenseFull, JoltOneHotK16,
        JoltOneHotK16Direct, JoltOneHotK16MultiChunk, JoltOneHotK16MultiChunkDirect,
        JoltOneHotK16W2R2, JoltOneHotK16W2R2Direct, JoltOneHotK16W4R2, JoltOneHotK16W4R2Direct,
        JoltOneHotK256, JoltOneHotK256Direct, JoltOneHotK256MultiChunk,
        JoltOneHotK256MultiChunkDirect, JoltOneHotK256W2R2, JoltOneHotK256W2R2Direct,
        JoltOneHotK256W4R2, JoltOneHotK256W4R2Direct,
    };
    use crate::planning::plan_schedule;
    use crate::{AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256};

    /// Native trace counts: 192/log_K + carry + bytecode and RAM chunks.
    /// Bytecode PCs fit u32; remapped u64 addresses have at most 61 word bits.
    /// K=16 is additionally bounded by the 64-column row mask.
    pub const K16_NUM_POLYS: &[usize] = &[51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64];
    pub const K256_NUM_POLYS: &[usize] = &[27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37];
    pub const K16_NUM_VARS: (usize, usize) = (16, 34);
    pub const K256_NUM_VARS: (usize, usize) = (33, 38);
    /// Bounded-dense advice and committed-program byte objects.
    pub const DENSE_NUM_VARS: (usize, usize) = (14, 34);

    /// First logical trace exponent using setup offloading.
    pub const RECURSIVE_TRACE_LOG_T_CUTOVER: usize = 21;
    pub const K16_COLUMN_VARIABLES: usize = 4;
    pub const K256_COLUMN_VARIABLES: usize = 8;

    /// Pure DP regeneration for `Cfg`; never consults an artifact.
    fn regen<Cfg: CommitmentConfig>(
        key: PolynomialGroupLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        plan_schedule::<Cfg>(&ScheduleLookupKey::single(key), &[])
    }

    fn regen_one_hot_k16(key: PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> {
        if key.num_vars() >= RECURSIVE_TRACE_LOG_T_CUTOVER + K16_COLUMN_VARIABLES {
            regen::<JoltOneHotK16>(key)
        } else {
            regen::<JoltOneHotK16Direct>(key)
        }
    }

    fn regen_one_hot_k256(key: PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> {
        if key.num_vars() >= RECURSIVE_TRACE_LOG_T_CUTOVER + K256_COLUMN_VARIABLES {
            regen::<JoltOneHotK256>(key)
        } else {
            regen::<JoltOneHotK256Direct>(key)
        }
    }

    fn regen_one_hot_k16_w2r2(key: PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> {
        if key.num_vars() >= RECURSIVE_TRACE_LOG_T_CUTOVER + K16_COLUMN_VARIABLES {
            regen::<JoltOneHotK16W2R2>(key)
        } else {
            regen::<JoltOneHotK16W2R2Direct>(key)
        }
    }

    fn regen_one_hot_k256_w2r2(key: PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> {
        if key.num_vars() >= RECURSIVE_TRACE_LOG_T_CUTOVER + K256_COLUMN_VARIABLES {
            regen::<JoltOneHotK256W2R2>(key)
        } else {
            regen::<JoltOneHotK256W2R2Direct>(key)
        }
    }

    fn regen_one_hot_k16_w4r2(key: PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> {
        if key.num_vars() >= RECURSIVE_TRACE_LOG_T_CUTOVER + K16_COLUMN_VARIABLES {
            regen::<JoltOneHotK16W4R2>(key)
        } else {
            regen::<JoltOneHotK16W4R2Direct>(key)
        }
    }

    fn regen_one_hot_k256_w4r2(key: PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> {
        if key.num_vars() >= RECURSIVE_TRACE_LOG_T_CUTOVER + K256_COLUMN_VARIABLES {
            regen::<JoltOneHotK256W4R2>(key)
        } else {
            regen::<JoltOneHotK256W4R2Direct>(key)
        }
    }

    fn regen_one_hot_k16_multi_chunk(
        key: PolynomialGroupLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        if key.num_vars() >= RECURSIVE_TRACE_LOG_T_CUTOVER + K16_COLUMN_VARIABLES {
            regen::<JoltOneHotK16MultiChunk>(key)
        } else {
            regen::<JoltOneHotK16MultiChunkDirect>(key)
        }
    }

    fn regen_one_hot_k256_multi_chunk(
        key: PolynomialGroupLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        if key.num_vars() >= RECURSIVE_TRACE_LOG_T_CUTOVER + K256_COLUMN_VARIABLES {
            regen::<JoltOneHotK256MultiChunk>(key)
        } else {
            regen::<JoltOneHotK256MultiChunkDirect>(key)
        }
    }

    fn reject_grouped(request: GroupedGenerationRequest) -> Result<FoldSchedule, AkitaError> {
        Err(AkitaError::InvalidSetup(format!(
            "jolt base families emit no grouped rows; refusing to plan {:?}",
            request.key()
        )))
    }

    /// Reachable scalar keys for one family grid.
    pub fn keys(
        num_polys: &[usize],
        (min_vars, max_vars): (usize, usize),
    ) -> Vec<PolynomialGroupLayout> {
        let mut keys = Vec::new();
        for &polys in num_polys {
            for num_vars in min_vars..=max_vars {
                let layout = OpeningClaimsLayout::new(num_vars, polys)
                    .and_then(|layout| layout.root_final_group_layout());
                if let Ok(key) = layout {
                    keys.push(key);
                }
            }
        }
        keys
    }

    /// Default production shapes plus the exact adapter, benchmark, and override fixtures.
    pub fn one_hot_keys(
        one_hot_k: usize,
        profile: AkitaOneHotChunkProfile,
    ) -> Result<Vec<PolynomialGroupLayout>, AkitaError> {
        let (widths, arities, fixtures): (_, _, &[(usize, usize)]) = match one_hot_k {
            AKITA_ONE_HOT_K16 => (
                K16_NUM_POLYS,
                K16_NUM_VARS,
                &[(12, 1), (12, 2), (16, 1), (25, 1)],
            ),
            AKITA_ONE_HOT_K256 => (
                K256_NUM_POLYS,
                K256_NUM_VARS,
                &[
                    (13, 27),
                    (14, 1),
                    (15, 1),
                    (16, 1),
                    (20, 1),
                    (25, 1),
                    (28, 27),
                    (29, 27),
                    (20, 29),
                ],
            ),
            other => {
                return Err(AkitaError::InvalidSetup(format!(
                    "unsupported one-hot K={other}"
                )))
            }
        };
        let mut admitted = keys(widths, arities);
        let fixtures = if profile == AkitaOneHotChunkProfile::Single {
            fixtures
        } else {
            &[(16, 1)]
        };
        admitted.extend(
            fixtures
                .iter()
                .map(|&(vars, polys)| PolynomialGroupLayout::new(vars, polys)),
        );
        Ok(admitted)
    }

    fn spec<Cfg: CommitmentConfig>(
        family_name: &'static str,
        keys: Vec<PolynomialGroupLayout>,
        regen: fn(PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError>,
        output_dir: PathBuf,
    ) -> Result<EmitSpec, AkitaError> {
        Ok(EmitSpec {
            family_name,
            policy: policy_of::<Cfg>(),
            source_contract: Cfg::committed_source_contract()?,
            keys,
            grouped_requests: Vec::new(),
            preplanned_scalar: Vec::new(),
            output_dir,
            regen,
            regen_group_batch: reject_grouped,
            ring_challenge_config: Cfg::ring_challenge_config,
        })
    }

    /// All base family specs, in emission order.
    ///
    /// Instance-specific grouped advice/program rows are planned during setup
    /// and folded into the exact catalog serialized with that verifier setup.
    pub fn family_specs(output_dir: PathBuf) -> Result<Vec<EmitSpec>, AkitaError> {
        Ok(vec![
            spec::<JoltOneHotK16>(
                JoltOneHotK16::schedule_family_name(),
                one_hot_keys(AKITA_ONE_HOT_K16, AkitaOneHotChunkProfile::Single)?,
                regen_one_hot_k16,
                output_dir.clone(),
            )?,
            spec::<JoltOneHotK256>(
                JoltOneHotK256::schedule_family_name(),
                one_hot_keys(AKITA_ONE_HOT_K256, AkitaOneHotChunkProfile::Single)?,
                regen_one_hot_k256,
                output_dir.clone(),
            )?,
            spec::<JoltOneHotK16W2R2>(
                JoltOneHotK16W2R2::schedule_family_name(),
                one_hot_keys(AKITA_ONE_HOT_K16, AkitaOneHotChunkProfile::Two)?,
                regen_one_hot_k16_w2r2,
                output_dir.clone(),
            )?,
            spec::<JoltOneHotK256W2R2>(
                JoltOneHotK256W2R2::schedule_family_name(),
                one_hot_keys(AKITA_ONE_HOT_K256, AkitaOneHotChunkProfile::Two)?,
                regen_one_hot_k256_w2r2,
                output_dir.clone(),
            )?,
            spec::<JoltOneHotK16W4R2>(
                JoltOneHotK16W4R2::schedule_family_name(),
                one_hot_keys(AKITA_ONE_HOT_K16, AkitaOneHotChunkProfile::Four)?,
                regen_one_hot_k16_w4r2,
                output_dir.clone(),
            )?,
            spec::<JoltOneHotK256W4R2>(
                JoltOneHotK256W4R2::schedule_family_name(),
                one_hot_keys(AKITA_ONE_HOT_K256, AkitaOneHotChunkProfile::Four)?,
                regen_one_hot_k256_w4r2,
                output_dir.clone(),
            )?,
            spec::<JoltOneHotK16MultiChunk>(
                JoltOneHotK16MultiChunk::schedule_family_name(),
                one_hot_keys(AKITA_ONE_HOT_K16, AkitaOneHotChunkProfile::Eight)?,
                regen_one_hot_k16_multi_chunk,
                output_dir.clone(),
            )?,
            spec::<JoltOneHotK256MultiChunk>(
                JoltOneHotK256MultiChunk::schedule_family_name(),
                one_hot_keys(AKITA_ONE_HOT_K256, AkitaOneHotChunkProfile::Eight)?,
                regen_one_hot_k256_multi_chunk,
                output_dir.clone(),
            )?,
            spec::<JoltDenseBounded>(
                JoltDenseBounded::schedule_family_name(),
                keys(&[1, 2], DENSE_NUM_VARS),
                regen::<JoltDenseBounded>,
                output_dir.clone(),
            )?,
            spec::<JoltDenseFull>(
                JoltDenseFull::schedule_family_name(),
                keys(&[1], DENSE_NUM_VARS),
                regen::<JoltDenseFull>,
                output_dir,
            )?,
        ])
    }
}
