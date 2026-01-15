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

### ABI / Memory Layout

TBD after measurement phase confirms:
- GT element size and limb layout
- Alignment requirements
- Aliasing rules

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
