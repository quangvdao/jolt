# BN254 GT Inlines for Dory in Jolt zkVM — Progress Tracker

## Goal

Reduce **zkVM virtual cycles** for Dory verification by introducing Jolt inlines for BN254 pairing target group (GT) operations:

- **GT multiplication** (group "add")
- **GT exponentiation** (group "scalar mul")

Primary target: Dory-heavy paths in the **recursive verifier** (guest runs verifier).

---

## Why This Matters

Dory combines commitments in GT as a scalar-weighted sum, which becomes GT exp + GT mul:

```rust
// jolt-core/src/poly/commitment/dory/commitment_scheme.rs L241-L248
coeffs
    .par_iter()
    .zip(commitments_vec.par_iter())
    .map(|(coeff, commitment)| {
        let ark_coeff = jolt_to_ark(coeff);
        ark_coeff * **commitment          // <-- GT exponentiation (scalar mul)
    })
    .reduce(ArkGT::identity, |a, b| a + b) // <-- GT multiplication (group add)
```

---

## High-Level Plan (Phases)

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Measurement & attribution | 🔲 Not started |
| 1 | Spec: instruction encoding + ABI + invariants | 🔲 Not started |
| 2 | Implement GT_MUL / GT_SQR inline sequences + tests | 🔲 Not started |
| 3 | Implement GT_EXP inline sequence + tests | 🔲 Not started |
| 4 | Integrate into verifier (Dory paths) + e2e proof tests | 🔲 Not started |
| 5 | (Optional) Deeper accelerators (Fq montmul, cyclotomic, multi-exp) | 🔲 Not started |

---

## Phase 0 — Measurement & Attribution

**Owner:** (assign next agent)

### 0.0 Ground Rules

- **Always measure in `--release`.**
- **Prevent optimizer distortion** with `core::hint::black_box` around values computed inside spans.
- **Use string literals for cycle labels.**
  - The cycle-tracking runtime keys markers by the label pointer address, not by string contents, so start/end must pass the *same pointer*.
  - Reference: `tracer/src/emulator/cpu.rs` L990–L1025

### 0.1 Deliverables

Produce a short report with:

| Item | Description |
|------|-------------|
| **A. End-to-end baseline** | Total "verification" cycles (virtual + RV64IMAC) for a representative recursive verification workload |
| **B. Stage attribution** | Virtual cycles for Dory-related stage(s) (e.g., Stage 8 / Dory batch opening verification) |
| **C. GT op attribution** | Number of GT exponentiations and GT multiplications; share of Dory stage cycles |
| **D. Microbenchmarks** | Virtual cycles per GT_MUL and GT_EXP (254-bit scalar) with overhead calibration |
| **E. Input characterization** | Exponent distribution; cyclotomic subgroup guarantee confirmation |

### 0.2 Tools Available In-Tree

#### Cycle Tracking Spans (recommended)

**API:**
```rust
use jolt::{start_cycle_tracking, end_cycle_tracking};

start_cycle_tracking("my_label");
// ... code to measure ...
end_cycle_tracking("my_label");
```

**References:**
- Implementation: `jolt-platform/src/cycle_tracking.rs` L8–L64
- Docs + examples: `book/src/usage/profiling/guest_profiling.md` L1–L72

**How cycles are computed:**
- Real cycles: `executed_instrs` delta
- Virtual cycles: `trace_len` delta
- Reference: `tracer/src/emulator/cpu.rs` L1015–L1021

#### Trace-Level Instruction Mix (optional)

Get a `ProgramSummary` of a run and count instruction occurrences:

```rust
// Build summary
let summary = program.trace_analyze::<F>(&inputs, &untrusted, &trusted);

// Count instructions
let counts: Vec<(&'static str, usize)> = summary.analyze::<F>();
```

**References:**
- `Program::trace_analyze`: `jolt-core/src/host/program.rs` L351–L366
- `ProgramSummary::analyze`: `jolt-core/src/host/analyze.rs` L24–L40
- Macro-generated `analyze_<fn>()`: `jolt-sdk/macros/src/lib.rs` L301–L346

### 0.3 Baseline: Run Existing Recursive Verification

Start from the existing recursion guest which already has a `"verification"` span:

**Reference:** `examples/recursion/guest/src/lib.rs` L29–L53

```rust
start_cycle_tracking("verification");
let verifier = RV64IMACVerifier::new(&verifier_preprocessing, proof, device, None, None);
let is_valid = verifier.is_ok_and(|verifier| verifier.verify().is_ok());
end_cycle_tracking("verification");
```

**Procedure:**
1. Pick a representative input (fixed proof bundle for reproducibility)
2. Run and record:
   - `"verification": … virtual cycles` output
   - Trace length printed by the host runner
3. Subdivide verification into stages (next section)

#### Baseline Numbers (trace-only, release, recursion/fibonacci, embed)

Run:
`RUST_LOG=info cargo run -p recursion --release trace --example fibonacci --embed`

**Deserialization** (span `"deserialization"`)
| Item | RV64IMAC cycles | Virtual cycles |
|------|-----------------|----------------|
| `deserialization (preprocessing + count)` | 13,529,574 | 20,433,445 |
| `deserialize preprocessing` | 13,529,523 | 20,433,364 |
| `deserialize count of proofs` | 29 | 59 |
| `deserialization (per proof, n=1)` | 10,266,182 | 11,097,885 |
| `deserialize proof` | 10,265,302 | 11,095,611 |
| `deserialize device` | 864 | 2,258 |

**Verification** (span `"verification"`)
| Item | RV64IMAC cycles | Virtual cycles | Notes |
|------|-----------------|----------------|-------|
| `verification (total)` | 1,078,773,999 | 1,170,607,347 | |
| `dory_opening_verify` | 673,694,170 | 730,024,753 | |
| `dory_gt_exp_ops` | 380,771,004 | 408,942,199 | homomorphic combine only |
| `dory_gt_mul_ops` | 1,349,687 | 1,387,759 | homomorphic combine only |
| `dory_gt_combine_counts` | - | - | exp=41, mul=40 |
| `dory_gt_exp_per_op` | 9,287,098 | 9,974,200 | RV/virt cycles per exp (380,771,004 / 41) |
| `dory_gt_mul_per_op` | 33,742 | 34,694 | RV/virt cycles per mul (1,349,687 / 40) |
| `dory_opening_gt_ops (derived)` | - | - | rounds=6, mul=74, exp=65 |
| `dory_opening_gt_cycles_estimate` | 606,158,269 | 650,890,353 | derived ops * per-op cost (estimate) |
| `verification_remainder` | 22,959,138 | 30,252,636 | verification minus opening + combine GT spans |

### 0.4 Add Stage-Level Spans Around Dory Verification

**Goal:** Isolate Dory's contribution.

**Suggested span placement:**

| Span Label | Where to Add |
|------------|--------------|
| `"verification_total"` | Outermost wrapper around full verify |
| `"stage8_dory"` | Around `verify_stage8` call |
| `"dory_verify_call"` | Around `dory::verify` call |
| `"dory_combine_commitments"` | Around GT combination loop |

**Reference entrypoints:**
- Stage 8: `jolt-core/src/zkvm/verifier.rs` L551–L588
- GT combine: `jolt-core/src/poly/commitment/dory/commitment_scheme.rs` L230–L249

### 0.5 GT Operation Attribution

You need both:
- **How many GT ops are executed** in Dory verify
- **How expensive each op is**

#### Current instrumentation (added)
- Cycle span around GT exponentiations: `"dory_gt_exp_ops"`
- Cycle span around GT multiplications (reduce): `"dory_gt_mul_ops"`
- Cycle span around Dory opening proof verify: `"dory_opening_verify"`
  - Implementation: `jolt-core/src/poly/commitment/dory/commitment_scheme.rs` (in `verify`)
 - Printed derived GT op counts for opening verify: `dory_opening_gt_ops (derived)`
   - Derived from proof `sigma` (rounds) using Dory verifier formulas

#### Option A: Instrument Call Sites (preferred if localized)

Create tiny wrappers:
```rust
fn gt_mul(a: &ArkGT, b: &ArkGT) -> ArkGT {
    GT_MUL_COUNT.fetch_add(1, Ordering::Relaxed);
    *a + *b
}

fn gt_exp(base: &ArkGT, scalar: &ArkFr) -> ArkGT {
    GT_EXP_COUNT.fetch_add(1, Ordering::Relaxed);
    *scalar * *base
}
```

#### Option B: Microbench Inside Guest (independent)

**Microbench design:**
```rust
#[jolt::provable]
fn gt_microbench(n: u32) -> [u8; 32] {
    let x = /* construct deterministic GT element */;
    let s = /* construct deterministic Fr scalar */;
    let mut acc = x.clone();

    start_cycle_tracking("gt_mul_loop");
    for _ in 0..n {
        acc = core::hint::black_box(acc * x);
    }
    end_cycle_tracking("gt_mul_loop");

    start_cycle_tracking("gt_exp_loop");
    for _ in 0..n {
        acc = core::hint::black_box(s * acc);
    }
    end_cycle_tracking("gt_exp_loop");

    /* return hash of acc to prevent dead code elimination */
}
```

**Calibration:**
- Run matched "empty loop" span to measure overhead
- Compute: `cycles_per_op = (virt_cycles(op_loop) - virt_cycles(overhead_loop)) / N`

**Parameters:**
- N = 256–4096 (large enough to drown out noise)
- Include edge cases: scalar = 0, 1, r-1

### 0.6 Final Measurement Artifact

Deliver a markdown note with:

- [ ] Workload description (what proof(s), what sizes, what config)
- [ ] Cycle span table (virtual cycles primary, RV64IMAC secondary)
- [ ] GT op counts (if Option A)
- [ ] Microbench results (Option B)
- [ ] Top 20 instruction counts from `ProgramSummary::analyze()`
- [ ] Conclusions:
  - Is GT_EXP dominant vs GT_MUL?
  - Do we need GT_SQR inline?
  - Is a multi-exp inline justified?
  - Any constraints discovered (e.g., cyclotomic subgroup assumptions)?

---

## Code Pointers / References

| What | Where |
|------|-------|
| Cycle tracking API | `jolt-platform/src/cycle_tracking.rs` L8–L64 |
| Guest profiling docs | `book/src/usage/profiling/guest_profiling.md` L1–L72 |
| Marker counting (real vs virtual) | `tracer/src/emulator/cpu.rs` L990–L1025 |
| Existing recursion guest spans | `examples/recursion/guest/src/lib.rs` L29–L53 |
| Stage 8 (Dory batch opening) | `jolt-core/src/zkvm/verifier.rs` L551–L588 |
| GT combine pattern | `jolt-core/src/poly/commitment/dory/commitment_scheme.rs` L230–L249 |
| Trace summarize | `jolt-core/src/host/program.rs` L351–L366 |
| Instruction mix | `jolt-core/src/host/analyze.rs` L24–L40 |

---

## Phase 1 — Specification (TODO)

### Instruction Encoding

| Inline | Opcode | funct7 | funct3 | Description |
|--------|--------|--------|--------|-------------|
| BN254_GT_MUL | 0x0B | 0x06 | 0x00 | GT group multiplication (a + b) |
| BN254_GT_SQR | 0x0B | 0x06 | 0x01 | GT group squaring (a + a) |
| BN254_GT_EXP | 0x0B | 0x06 | 0x02 | GT exponentiation (s * a) |

### GT exponentiation semantics (arkworks / Dory)

This section specifies the **exact semantics** of “GT exponentiation” as currently implemented in the Dory PCS backend used by Jolt.

#### Types / groups

- **Scalar field**: BN254 scalar field `Fr` with modulus

```1:10:/Users/quang.dao/.cargo/git/checkouts/arkworks-algebra-55219ebe4db9d51c/76bb3a4/curves/bn254/src/fields/fr.rs
use ark_ff::fields::{Fp256, MontBackend, MontConfig};

#[derive(MontConfig)]
#[modulus = "21888242871839275222246405745257275088548364400416034343698204186575808495617"]
#[generator = "5"]
#[small_subgroup_base = "3"]
#[small_subgroup_power = "2"]
pub struct FrConfig;
pub type Fr = Fp256<MontBackend<FrConfig, 4>>;
```

- **Target group**: BN254 pairing target group `GT`, represented as the extension field element `Fq12` (multiplicative group).

#### What operation “GT exp” means

At the Dory layer, GT is treated as a `Group` where:
- **identity** is \(1 \in Fq12\)
- **group “add”** is **field multiplication** in \(Fq12\)
- **group “neg”** is **field inverse** in \(Fq12\)
- **group “scale”** (a.k.a. “scalar mul”) is **field exponentiation** in \(Fq12\)

This is implemented in the Dory arkworks backend (`dory-pcs`) as:

```206:225:/Users/quang.dao/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/dory-pcs-0.1.0/src/backends/arkworks/ark_group.rs
impl Group for ArkGT {
    type Scalar = ArkFr;

    fn identity() -> Self {
        ArkGT(Fq12::one())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArkGT(self.0 * rhs.0)
    }

    fn neg(&self) -> Self {
        ArkGT(ArkField::inverse(&self.0).expect("GT inverse"))
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        ArkGT(self.0.pow(k.0.into_bigint()))
    }
}
```

And in Jolt’s homomorphic combine path, GT exponentiation is invoked via `ArkFr * ArkGT`:

```279:292:/Users/quang.dao/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/dory-pcs-0.1.0/src/backends/arkworks/ark_group.rs
impl Mul<ArkGT> for ArkFr {
    type Output = ArkGT;
    fn mul(self, rhs: ArkGT) -> ArkGT {
        ArkGT(rhs.0.pow(self.0.into_bigint()))
    }
}

impl<'a> Mul<&'a ArkGT> for ArkFr {
    type Output = ArkGT;
    fn mul(self, rhs: &'a ArkGT) -> ArkGT {
        ArkGT(rhs.0.pow(self.0.into_bigint()))
    }
}
```

Jolt calls that here (this is the “homomorphic combine only” GT exp measurement):

```264:301:/Users/quang.dao/Documents/SNARKs/jolt/jolt-core/src/poly/commitment/dory/commitment_scheme.rs
    // ... snip ...
    let exp_terms: Vec<ArkGT> = coeffs
        .par_iter()
        .zip(commitments_vec.par_iter())
        .map(|(coeff, commitment)| {
            let ark_coeff = jolt_to_ark(coeff);
            core::hint::black_box(ark_coeff) * core::hint::black_box(**commitment)
        })
        .collect();
    // ... snip ...
```

#### Exponent conversion (scalar → integer)

The exponent is the scalar field element `Fr` interpreted as an integer \(e \in [0, r-1]\) via:
- `e = k.into_bigint()` (arkworks `PrimeField::into_bigint`)
- representation: **4× `u64` limbs**, **least significant limb first** (`MontBackend<..., 4>`)

So the GT exp computes \(g^e\) (not \(g^{e \bmod (q^{12}-1)}\) explicitly; field exponentiation semantics imply this).

#### Exact exponentiation algorithm (`pow`)

`Fq12::pow(exp)` uses the default `ark_ff::Field::pow` implementation: **left-to-right square-and-multiply**, iterating exponent bits from **MSB → LSB**, skipping leading zeros, where `exp` is provided as `u64` limbs (LS limb first).

```316:330:/Users/quang.dao/.cargo/git/checkouts/arkworks-algebra-55219ebe4db9d51c/76bb3a4/ff/src/fields/mod.rs
    /// Returns `self^exp`, where `exp` is an integer represented with `u64` limbs,
    /// least significant limb first.
    #[must_use]
    fn pow<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let mut res = Self::one();

        for i in crate::BitIteratorBE::without_leading_zeros(exp) {
            res.square_in_place();

            if i {
                res *= self;
            }
        }
        res
    }
```

#### Edge cases (must match)

- **Scalar = 0**: exponent iterator is empty → returns `1` (`GT::identity`)
- **Scalar = 1**: returns `g`
- **Scalar = r-1**: returns `g^(r-1)` (same as `g^{-1}` iff `g^r = 1`, which holds for subgroup elements, but the implementation is purely `pow`)

### Inline contract (BN254_GT_EXP)

This is the **proposed inline ABI contract** for `BN254_GT_EXP`, matching the conventions used by existing Jolt core inlines (e.g. `BIGINT256_MUL`).

#### Instruction encoding

- **opcode**: `0x0B` (Jolt core inlines) (see `INLINE_OPCODE`)
- **funct7**: `0x06` (BN254 family; reserved for this workstream)
- **funct3**: `0x02` (GT exponentiation)

The inline is encoded using the assembler form:

```text
.insn r opcode, funct3, funct7, rd, rs1, rs2
```

But **Jolt interprets the R-format `rd` field as an extra source register `rs3`**, and **does not permit the inline to modify any architectural registers**. This mapping is explicit in the tracer:

```1:9:/Users/quang.dao/Documents/SNARKs/jolt/tracer/src/instruction/format/format_inline.rs
//! FormatInline writes results to memory pointed by `rs3` (or rs1/rs2), but never
//! modifies the register values themselves - only the memory they reference.
//! Note: SDKs use FormatR in assembly code ... but are parsed as FormatInline.
```

```74:80:/Users/quang.dao/Documents/SNARKs/jolt/tracer/src/instruction/format/format_inline.rs
    fn parse(word: u32) -> Self {
        FormatInline {
            rs3: ((word >> 7) & 0x1f) as u8,  // [11:7]  (R-format rd)
            rs1: ((word >> 15) & 0x1f) as u8, // [19:15]
            rs2: ((word >> 20) & 0x1f) as u8, // [24:20]
        }
    }
```

#### Register / pointer contract

At runtime, the three registers carry 64-bit guest pointers:

- **rs1**: `exp_ptr` — pointer to exponent `e` as 4×`u64` limbs (little-endian)
- **rs2**: `base_ptr` — pointer to base `g` as 48×`u64` limbs (see layout below)
- **rs3** (encoded in `rd` field): `out_ptr` — pointer to output `g^e` as 48×`u64` limbs

This follows the same “(rs1, rs2) inputs; (rd/rs3) output pointer” pattern used by the existing `BIGINT256_MUL` inline:

```24:49:/Users/quang.dao/Documents/SNARKs/jolt/jolt-inlines/bigint/src/multiplication/sdk.rs
/// * `a` - Pointer to 4 u64 words (32 bytes) for first operand
/// * `b` - Pointer to 4 u64 words (32 bytes) for second operand
/// * `result` - Pointer to 8 u64 words (64 bytes) where result will be written
#[cfg(not(feature = "host"))]
pub unsafe fn bigint256_mul_inline(a: *const u64, b: *const u64, result: *mut u64) {
    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
        rd = in(reg) result,  // rd - output address
        rs1 = in(reg) a,      // rs1 - first operand address
        rs2 = in(reg) b,      // rs2 - second operand address
        options(nostack)
    );
}
```

#### Memory layout

All words are **little-endian limbs** (`u64`), and all pointers must be **8-byte aligned**.

- **Exponent (`exp_ptr`)**: 4×`u64` limbs, representing the integer exponent \(e\) in base \(2^{64}\), least-significant limb first:
  - \(e = \sum_{i=0}^{3} exp[i]\cdot 2^{64i}\)
  - **Bit numbering**: bit 0 is LSB of `exp[0]`; bit 255 is MSB of `exp[3]`.
  - The inline must iterate bits **MSB→LSB skipping leading zeros** (same as arkworks `Field::pow` via `BitIteratorBE::without_leading_zeros`).

- **GT element (`base_ptr`, `out_ptr`)**: 48×`u64` limbs, representing BN254 `Fq12` in the canonical tower order:
  - `Fq12 = Fp12(c0: Fq6, c1: Fq6)`
  - `Fq6 = (c0: Fq2, c1: Fq2, c2: Fq2)`
  - `Fq2 = (c0: Fq, c1: Fq)`
  - `Fq` is 256-bit, stored as 4×`u64` limbs (little-endian).

Flattened order (12 `Fq` elements × 4 limbs each = 48 limbs):

1. `c0.c0.c0` (Fq) limbs[0..4)
2. `c0.c0.c1`
3. `c0.c1.c0`
4. `c0.c1.c1`
5. `c0.c2.c0`
6. `c0.c2.c1`
7. `c1.c0.c0`
8. `c1.c0.c1`
9. `c1.c1.c0`
10. `c1.c1.c1`
11. `c1.c2.c0`
12. `c1.c2.c1`

Each `Fq` is in **Montgomery representation**, matching arkworks’ internal field representation used by multiplication/squaring in `Fq12`.

#### Functional correctness requirement (must match)

Given `exp_ptr` encoding integer \(e\) and `base_ptr` encoding field element \(g \in Fq12\), the inline must write to `out_ptr` the field exponentiation:

\[
out \gets g^e \in Fq12
\]

with exponentiation semantics matching arkworks `Field::pow` (square-and-multiply, MSB→LSB, skip leading zeros). See:

```316:330:/Users/quang.dao/.cargo/git/checkouts/arkworks-algebra-55219ebe4db9d51c/76bb3a4/ff/src/fields/mod.rs
fn pow<S: AsRef<[u64]>>(&self, exp: S) -> Self {
    let mut res = Self::one();
    for i in crate::BitIteratorBE::without_leading_zeros(exp) {
        res.square_in_place();
        if i { res *= self; }
    }
    res
}
```

#### Aliasing / safety rules

- **Pointer validity**:
  - `exp_ptr` must be readable for 32 bytes.
  - `base_ptr` must be readable for 384 bytes.
  - `out_ptr` must be writable for 384 bytes.
- **Aliasing**:
  - `out_ptr` may equal `base_ptr` (in-place update permitted) **in principle**.
    - **Current status**: the in-progress `BN254_GT_EXP` inline implementation rejects `out_ptr == base_ptr`
      (it asserts `base_ptr != out_ptr`) because it uses `out_ptr[..384]` as scratch during computation.
      Supporting in-place aliasing will require an alias-safe schedule and/or additional scratch strategy.
  - `out_ptr` must not overlap `exp_ptr`.
- **Side effects**: the inline must not modify memory outside `out_ptr[..384]`.

#### Inline-sequence hygiene requirements

The expanded inline sequence must:
- use only memory + virtual registers for scratch,
- and **zero all virtual registers before returning** (enforced by the inline builder finalization in the tracer).

```142:150:/Users/quang.dao/Documents/SNARKs/jolt/tracer/src/utils/inline_helpers.rs
    /// Finalize inline instructions by zeroing virtual registers, then calling finalize.
    pub fn finalize_inline(mut self) -> Vec<Instruction> {
        let register = self.allocator.get_registers_for_reset();
        // Zero inline virtual registers using ADDI rd, x0, 0
        for reg in register {
            self.emit_i::<ADDI>(reg, 0, 0);
        }
        self.finalize()
    }
```

### ABI / Memory Layout

See **Inline contract (BN254_GT_EXP)** above. (We can define analogous contracts for `BN254_GT_MUL` / `BN254_GT_SQR` after choosing whether they should operate on the same 48×`u64` `Fq12` layout and whether multiplication is `out = a * b` or `out = a ⊗ b` with an `out_ptr`.)

### Mathematical Preconditions

TBD:
- Cyclotomic subgroup membership assumed?
- Edge case handling (identity, zero scalar)

---

## Phase 2–5 — Implementation (TODO)

To be planned after Phase 0 measurement results.

---

## Security / Soundness Verification Plan

### Functional Correctness
- [ ] Differential tests against arkworks reference (thousands of random cases)
- [ ] Edge cases: identity, zero scalar, max scalar, etc.

### Algebraic Property Tests
- [ ] Associativity spot-checks
- [ ] Exponent laws: a^{x+y} = a^x * a^y, (a^x)^y = a^{xy}

### Trace-Level Tests
- [ ] Generate `.joltinline` trace, verify matches regenerated sequences
- [ ] Negative tests: if advice used, verify wrong advice spoils proof

### End-to-End Proof Tests
- [ ] Minimal guest program with GT ops → produce and verify proof
- [ ] Regression tests on representative Dory verification traces
