# Jolt Verifier Cycle Count Analysis

## Summary

**Total Verification: 2,165M total cycles** (1,038M RV64IMAC + 1,128M virtual)

The dominant cost is **Dory PCS verification (stage 8)** at **2,113M total cycles (97.6%)**, with:
- **Reduce-and-fold rounds**: **1,169M total cycles (54.0%)**
- **GT scalar multiplications** for joint commitment: **792M total cycles (36.6%)**
- **Final verification (multi-pairing)**: **140M total cycles (6.5%)**

## Measured Cycle Breakdown

### Top-Level Breakdown

| Operation | RV64IMAC | Virtual | **Total** | % of Total |
|-----------|----------|---------|-----------|------------|
| **Total Verification** | 1,037,566,327 | 1,127,723,671 | **2,165,289,998** | **100%** |
| deserialize preprocessing | 13,526,078 | 20,428,609 | 33,954,687 | 1.6% |
| deserialize proof | 10,259,205 | 11,107,130 | 21,366,335 | 1.0% |
| deserialize device | 850 | 2,246 | 3,096 | ~0% |
| **Jolt Verify Total** | 1,037,277,689 | 1,127,246,839 | **2,164,524,528** | **~100%** |

### Jolt Verification Stages

| Stage | RV64IMAC | Virtual | **Total** | % of Total |
|-------|----------|---------|-----------|------------|
| Preamble (commitment hashing) | 831,707 | 1,501,923 | 2,333,630 | 0.11% |
| **Stage 1** (Spartan outer) | 1,038,485 | 1,430,935 | 2,469,420 | 0.11% |
| **Stage 2** (Product/RAM) | 1,627,194 | 2,157,633 | 3,784,827 | 0.17% |
| **Stage 3** (Spartan shift) | 503,464 | 681,375 | 1,184,839 | 0.05% |
| **Stage 4** (Registers R/W) | 1,648,014 | 1,950,832 | 3,598,846 | 0.17% |
| **Stage 5** (Registers val) | 5,834,907 | 7,602,251 | 13,437,158 | 0.62% |
| **Stage 6** (Bytecode/Booleanity) | 9,950,950 | 12,890,691 | 22,841,641 | 1.05% |
| **Stage 7** (Hamming weight) | 781,597 | 957,292 | 1,738,889 | 0.08% |
| **Stage 8 (Dory PCS)** | 1,015,061,305 | 1,098,073,839 | **2,113,135,144** | **97.6%** |

### Sumcheck Operations (Across All Stages)

| Stage | Input Claims | Rounds Verify | Expected Output | **Stage Total** |
|-------|--------------|---------------|-----------------|-----------------|
| Stage 1 | 2,082 | 609,416 | 1,304,515 | 1,941,961 |
| Stage 2 | 16,064 | 1,420,701 | 1,559,141 | 3,125,406 |
| Stage 3 | 22,465 | 541,559 | 483,025 | 1,138,921 |
| Stage 4 | 14,391 | 1,015,040 | 189,658 | 1,302,535 |
| Stage 5 | 14,533 | 8,279,784 | 5,023,794 | 13,400,667 |
| Stage 6 | 32,305 | 1,401,128 | 11,955,200 | 13,556,290 |
| Stage 7 | 86,977 | 219,686 | 1,130,172 | 1,547,651 |
| **Total** | **188,817** | **13,487,314** | **21,645,505** | **36,013,431** |

**Sumcheck totals: ~36M total cycles (1.7% of total)**

### Expected Output Claims Breakdown (cache_openings vs compute_expected_claim)

The `expected_output_claims` phase has two components:
- **cache_openings**: Transcript operations (hashing commitments, challenges)
- **compute_expected_claim**: MLE evaluations at sumcheck challenge point

| Stage | cache_openings (RV64IMAC) | cache_openings (Virtual) | compute_expected_claim (RV64IMAC) | compute_expected_claim (Virtual) |
|-------|---------------------------|--------------------------|-----------------------------------|----------------------------------|
| Stage 1 | 196,971 | 300,513 | 374,833 | 431,401 |
| Stage 2 | 96,332 | 145,064 | 614,047 | 699,115 |
| Stage 3 | 87,370 | 132,858 | 128,096 | 132,354 |
| Stage 4 | 53,151 | 79,324 | 25,904 | 28,302 |
| Stage 5 | 304,487 | 459,308 | 2,110,001 | 2,147,589 |
| Stage 6 | 454,962 | 675,207 | 4,508,471 | 6,313,939 |
| Stage 7 | 219,642 | 326,132 | 287,799 | 295,838 |
| **Total** | **1,412,915** | **2,118,406** | **8,049,151** | **10,048,538** |

**Key Finding**: The expected output claims cost is dominated by **MLE evaluations** (`compute_expected_claim` = 18.1M total), NOT transcript hashing (`cache_openings` = 3.5M total).

**Stage 6 Deep Dive**: The most expensive `compute_expected_claim` in Stage 6 costs **4,357,111 RV64IMAC + 6,149,220 virtual = 10.5M total cycles** for a single sumcheck instance (instruction lookups read RAF).

### Stage 8: Dory PCS Verification (THE BOTTLENECK)

| Operation | RV64IMAC | Virtual | **Total** | % of Stage 8 |
|-----------|----------|---------|-----------|--------------|
| dory_init_context | 164 | 312 | 476 | ~0% |
| dory_collect_claims | 250,456 | 376,444 | 626,900 | 0.03% |
| dory_build_commitments_map | 90,000 | 111,368 | 201,368 | 0.01% |
| **dory_compute_joint_commitment** | 382,197,138 | 410,441,350 | **792,638,488** | **37.5%** |
| dory_verify_prep | 320 | 329 | 649 | ~0% |
| dory_vmv_transcript | 38,414 | 58,238 | 96,652 | ~0% |
| dory_verifier_state_init | 4,642,119 | 5,374,811 | 10,016,930 | 0.47% |
| **dory_reduce_fold_rounds** | 560,554,783 | 608,351,796 | **1,168,906,579** | **55.3%** |
| **dory_verify_final** | 67,233,356 | 73,295,134 | **140,528,490** | **6.6%** |
| **Total dory_eval_proof_verify** | 632,508,687 | 687,109,992 | **1,319,618,679** | **62.4%** |

### Dory Final Verification Breakdown

| Operation | RV64IMAC | Virtual | **Total** | % of Final |
|-----------|----------|---------|-----------|------------|
| dory_final_field_ops (γ⁻¹, d⁻¹) | 63,219 | 74,715 | 137,934 | 0.10% |
| **dory_final_rhs_gt_ops** | 33,558,800 | 36,064,172 | **69,622,972** | **49.5%** |
| **dory_final_pairing_prep** (G1/G2 mults) | 17,286,524 | 19,781,060 | **37,067,584** | **26.4%** |
| dory_bn254_multi_miller_loop | 7,270,979 | 7,516,038 | 14,787,017 | 10.5% |
| dory_bn254_final_exponentiation | 5,042,291 | 5,160,291 | 10,202,582 | 7.3% |
| **dory_final_multi_pairing** | 16,321,929 | 17,366,896 | **33,688,825** | **24.0%** |

## Key Insights

### 1. GT Scalar Multiplications are the Bottleneck (~55%)

The **reduce-and-fold rounds** take **1,169M total cycles** (55% of total). Each of the ~11 rounds involves:
- GT scalar multiplications for batching
- Transcript hashing for challenge generation
- Point updates

### 2. Joint Commitment Computation is Expensive (~37%)

Computing `dory_compute_joint_commitment` takes **793M total cycles** (37% of total). This is:
```
joint_commitment = Σ γᵢ · Cᵢ
```
where each `Cᵢ` is a GT element and there are ~30+ commitments.

**This involves ~30+ GT scalar multiplications** which are extremely expensive on RISC-V.

### 3. Sumcheck is Relatively Cheap (~1.7%)

All sumcheck operations combined take only **36M total cycles**:
- Input claims: 189K total cycles
- Rounds verification: 13.5M total cycles  
- Expected output claims: 21.6M total cycles
  - Of which **cache_openings (transcript)**: 3.5M total cycles
  - Of which **compute_expected_claim (MLE evals)**: 18.1M total cycles

**The expected output claims are dominated by MLE evaluations, not transcript hashing.**

### 4. Multi-Pairing is Reasonable (~1.6%)

The final multi-pairing (3 Miller loops + 1 final exponentiation) takes **33.7M total cycles**:
- Miller loops: 14.8M total cycles
- Final exponentiation: 10.2M total cycles

### 5. Deserialization is Minor (~2.5%)

Deserializing preprocessing + proofs takes ~55M total cycles, a small fraction of total.

## Answer to the Original Question

> "Does the 140 million cycle count include G_T scalar mults for homomorphic combining of signatures? I assumed those were offloaded to P using sum-check+Hyrax."

**Yes, the GT scalar multiplications ARE included and are the dominant cost!**

The current implementation does **NOT** offload GT combining to the prover via sum-check + Hyrax. Instead:

1. **Joint commitment computation** (`dory_compute_joint_commitment`): The verifier computes `Σ γᵢ · Cᵢ` directly with **30+ GT scalar multiplications** costing **793M total cycles**.

2. **Reduce-and-fold rounds** (`dory_reduce_fold_rounds`): Each of the ~11 Dory reduction rounds performs GT operations costing **1,169M total cycles**.

These two operations alone account for **1,962M total cycles (91%)** of the total verification cost.

## Recommendations

To reduce cycle count to the theoretical ~36M estimate (sumcheck + field ops), consider:

1. **Offload GT combining to prover**: Use sum-check + inner product arguments to replace direct GT scalar multiplications with field operations that the verifier can check.

2. **Reduce Dory rounds**: The 11 Dory reduction rounds are expensive. Consider batch opening or alternative commitment schemes.

3. **Use a different PCS**: Consider schemes with cheaper verifier operations (e.g., FRI-based schemes with only field operations).

---

## How to Reproduce

### Step 1: Generate Proof (if not already done)

```bash
cargo run --release -p recursion generate --example fibonacci
```

### Step 2: Run Trace with Cycle Tracking

```bash
RUST_LOG=info cargo run --release -p recursion -- trace --example fibonacci --embed 2>&1 | tee trace_output.txt
```

### Step 3: Extract Cycle Counts

The trace output contains lines like:
```
INFO trace: tracer::emulator::cpu: "label": X RV64IMAC cycles, Y virtual cycles
```

---

## Minimal Diff for Cycle Tracking Instrumentation

The following changes enable the cycle tracking. The key pattern is using `CycleSpan::new("label")` as an RAII guard.

### 1. `third-party/dory/src/lib.rs` — Add cycle_tracking module

```rust
pub mod cycle_tracking;
```

### 2. `third-party/dory/src/cycle_tracking.rs` — New file

```rust
//! Cycle tracking markers for Jolt's RISC-V emulator.

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
mod riscv_specific {
    const JOLT_CYCLE_TRACK_ECALL_NUM: u32 = 0xC7C1E;
    const JOLT_CYCLE_MARKER_START: u32 = 1;
    const JOLT_CYCLE_MARKER_END: u32 = 2;

    #[inline(always)]
    pub(super) fn start_cycle_tracking(marker_id: &str) {
        unsafe {
            core::arch::asm!(
                ".word 0x00000073",
                in("x10") JOLT_CYCLE_TRACK_ECALL_NUM,
                in("x11") marker_id.as_ptr() as u32,
                in("x12") marker_id.len() as u32,
                in("x13") JOLT_CYCLE_MARKER_START,
                options(nostack, nomem, preserves_flags)
            );
        }
    }

    #[inline(always)]
    pub(super) fn end_cycle_tracking(marker_id: &str) {
        unsafe {
            core::arch::asm!(
                ".word 0x00000073",
                in("x10") JOLT_CYCLE_TRACK_ECALL_NUM,
                in("x11") marker_id.as_ptr() as u32,
                in("x12") marker_id.len() as u32,
                in("x13") JOLT_CYCLE_MARKER_END,
                options(nostack, nomem, preserves_flags)
            );
        }
    }
}

#[inline(always)]
#[allow(unused_variables)]
pub fn start_cycle_tracking(marker_id: &str) {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    riscv_specific::start_cycle_tracking(marker_id);
}

#[inline(always)]
#[allow(unused_variables)]
pub fn end_cycle_tracking(marker_id: &str) {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    riscv_specific::end_cycle_tracking(marker_id);
}

#[must_use]
pub struct CycleSpan<'a> {
    label: &'a str,
}

impl<'a> CycleSpan<'a> {
    #[inline(always)]
    pub fn new(label: &'a str) -> Self {
        start_cycle_tracking(label);
        Self { label }
    }
}

impl Drop for CycleSpan<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        end_cycle_tracking(self.label);
    }
}
```

### 3. `third-party/dory/src/reduce_and_fold.rs` — Instrument verify_final

Add at the top of `verify_final` function (around line 544):

```rust
use crate::cycle_tracking::CycleSpan;

// In verify_final():
let _cycle_prep = CycleSpan::new("dory_final_field_ops");
// ... field inversions ...
drop(_cycle_prep);

let _cycle_rhs = CycleSpan::new("dory_final_rhs_gt_ops");
// ... GT scalar mults for RHS ...
drop(_cycle_rhs);

let _cycle_pair_prep = CycleSpan::new("dory_final_pairing_prep");
// ... G1/G2 scalar mults ...
drop(_cycle_pair_prep);

let _cycle_pairing = CycleSpan::new("dory_final_multi_pairing");
let lhs = E::multi_pair(&[p1_g1, p2_g1, p3_g1], &[p1_g2, p2_g2, p3_g2]);
drop(_cycle_pairing);
```

### 4. `third-party/dory/src/evaluation_proof.rs` — Instrument verify_evaluation_proof

```rust
use crate::cycle_tracking::CycleSpan;

pub fn verify_evaluation_proof<...>(...) -> Result<(), DoryError> {
    let _cycle_total = CycleSpan::new("dory_eval_proof_verify_total");
    
    // ... existing code ...
    
    let _cycle_vmv = CycleSpan::new("dory_vmv_transcript");
    // ... VMV transcript operations ...
    drop(_cycle_vmv);
    
    let _cycle_init = CycleSpan::new("dory_verifier_state_init");
    // ... verifier state init ...
    drop(_cycle_init);
    
    let _cycle_rounds = CycleSpan::new("dory_reduce_fold_rounds");
    for round in 0..num_rounds {
        // ... existing round loop ...
    }
    drop(_cycle_rounds);
    
    let _cycle_final = CycleSpan::new("dory_verify_final");
    verifier_state.verify_final(&proof.final_message, &gamma, &d)
}
```

### 5. `third-party/dory/src/backends/arkworks/ark_pairing.rs` — Instrument multi_pair

```rust
use crate::cycle_tracking::CycleSpan;

fn multi_pair(g1_elements: &[Self::G1], g2_elements: &[Self::G2]) -> Self::GT {
    let _ml = CycleSpan::new("dory_bn254_multi_miller_loop");
    let ml_result = ArkBn254::multi_miller_loop(prepared_g1, prepared_g2);
    drop(_ml);
    
    let _fe = CycleSpan::new("dory_bn254_final_exponentiation");
    let result = ArkBn254::final_exponentiation(ml_result).unwrap();
    drop(_fe);
    
    result.0
}
```

### 6. `jolt-core/src/zkvm/verifier.rs` — Instrument verification stages

Add at the top of each file using cycle tracking:

```rust
use jolt_platform::cycle_tracking::{end_cycle_tracking, start_cycle_tracking};

struct CycleSpan<'a> { label: &'a str }
impl<'a> CycleSpan<'a> {
    #[inline(always)]
    fn new(label: &'a str) -> Self {
        start_cycle_tracking(label);
        Self { label }
    }
}
impl Drop for CycleSpan<'_> {
    #[inline(always)]
    fn drop(&mut self) { end_cycle_tracking(self.label); }
}
```

Then wrap each stage:

```rust
pub fn verify(mut self) -> Result<(), anyhow::Error> {
    let _cycle_total = CycleSpan::new("jolt_verify_total");
    
    {
        let _cycle = CycleSpan::new("jolt_verify_preamble");
        // ... preamble code ...
    }
    
    {
        let _cycle = CycleSpan::new("jolt_verify_stage1");
        self.verify_stage1()?;
    }
    // ... repeat for stages 2-8 ...
}
```

### 7. `jolt-core/src/subprotocols/sumcheck.rs` — Instrument sumcheck verification

```rust
use jolt_platform::cycle_tracking::{end_cycle_tracking, start_cycle_tracking};
// ... CycleSpan helper as above ...

pub fn verify<...>(...) -> Result<Vec<F::Challenge>, ProofVerifyError> {
    let _cycle_sumcheck = CycleSpan::new("batched_sumcheck_verify");
    
    let claim: F = {
        let _cycle = CycleSpan::new("sumcheck_input_claims");
        // ... input claims code ...
    };
    
    let (output_claim, r_sumcheck) = {
        let _cycle = CycleSpan::new("sumcheck_rounds_verify");
        proof.verify(claim, max_num_rounds, max_degree, transcript)?
    };
    
    let expected_output_claim: F = {
        let _cycle = CycleSpan::new("sumcheck_expected_output_claims");
        sumcheck_instances.iter().zip(batching_coeffs.iter()).map(|(sumcheck, coeff)| {
            // ...
            {
                let _c = CycleSpan::new("sumcheck_cache_openings");
                sumcheck.cache_openings(...);
            }
            let claim = {
                let _c = CycleSpan::new("sumcheck_compute_expected_claim");
                sumcheck.expected_output_claim(...)
            };
            claim * coeff
        }).sum()
    };
    // ...
}
```

### 8. `jolt-core/src/poly/commitment/dory/commitment_scheme.rs` — Instrument Dory PCS verify

```rust
fn verify<ProofTranscript: Transcript>(...) -> Result<(), ProofVerifyError> {
    let _cycle = CycleSpan::new("dory_verify_eval_proof");
    // ... existing verify code ...
}
```

### 9. `Cargo.toml` (workspace) — Use local dory

```toml
dory = { package = "dory-pcs", path = "./third-party/dory", default-features = false }
```

### 10. `jolt-core/Cargo.toml` — Configure dory features

```toml
[features]
prover = [
    "minimal",
    # ...
    "dory/arkworks",
    "dory/cache",
    "dory/disk-persistence"
]
minimal = ["ark-ec/std", "ark-ff/std", "ark-std/std", "ark-ff/asm", "rayon", "dory/arkworks"]

[dependencies]
dory.workspace = true
```

### 11. `jolt-core/src/poly/commitment/dory/dory_globals.rs` — Gate cache behind prover

```rust
#[cfg(feature = "prover")]
use dory::backends::arkworks::{init_cache, is_cached, ArkG1, ArkG2};

#[cfg(feature = "prover")]
pub fn init_prepared_cache(g1_vec: &[ArkG1], g2_vec: &[ArkG2]) {
    if !is_cached() { init_cache(g1_vec, g2_vec); }
}
```
