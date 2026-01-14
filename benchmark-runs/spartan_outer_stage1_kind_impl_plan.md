# Spartan outer Stage 1: “protocol selectable” proof (Approach 1) — Implementation Plan
**Status:** design doc / handoff for implementation  
**Goal:** allow end-to-end Jolt proving **and verification** to run Stage 1 using either:
- **Current protocol**: *uni-skip first round + remainder sumcheck* (existing)
- **Full-outer protocols (Option B)**: *baseline / naive / round-batched* (no uni-skip split)

**Constraints / assumptions:**
- **No backward compatibility needed** for old proofs.
- Default should remain the current behavior (uni-skip + remainder) unless overridden.
- This is intended for benchmarking first; correctness must still hold (verifier must accept).

---

## Why this change is needed (current state)

Today Stage 1 is hard-coded in both proof format and verifier:
- Proof contains `stage1_uni_skip_first_round_proof` and `stage1_sumcheck_proof`
- Verifier always runs `verify_stage1_uni_skip(...)` then verifies `OuterRemainingSumcheckVerifier`

See:

```28:48:jolt-core/src/zkvm/proof_serialization.rs
pub struct JoltProof<...> {
    pub stage1_uni_skip_first_round_proof: UniSkipFirstRoundProof<F, FS>,
    pub stage1_sumcheck_proof: SumcheckInstanceProof<F, FS>,
    ...
}
```

```213:237:jolt-core/src/zkvm/verifier.rs
fn verify_stage1(&mut self) -> Result<(), anyhow::Error> {
    let uni_skip_params = verify_stage1_uni_skip(
        &self.proof.stage1_uni_skip_first_round_proof,
        &self.spartan_key,
        &mut self.opening_accumulator,
        &mut self.transcript,
    )?;

    let spartan_outer_remaining = OuterRemainingSumcheckVerifier::new(
        self.spartan_key,
        self.proof.trace_length,
        uni_skip_params,
        &self.opening_accumulator,
    );

    BatchedSumcheck::verify(
        &self.proof.stage1_sumcheck_proof,
        vec![&spartan_outer_remaining],
        &mut self.opening_accumulator,
        &mut self.transcript,
    )?;
    Ok(())
}
```

This blocks “Option B” because full-outer variants do not have the same Stage 1 structure.

---

## High-level design

Replace the two fixed Stage 1 proof fields with:

1) A **proof-level tag** that says which Stage 1 protocol was used
2) A **tagged Stage 1 proof payload** that carries either:
   - (uni-skip proof + remainder sumcheck proof), or
   - (full-outer sumcheck proof)

Then:
- **Prover** chooses Stage 1 protocol based on config/env var, generates the matching Stage 1 proof variant.
- **Verifier** matches the tag and runs the matching Stage 1 verification logic.

---

## Concrete data model changes

### New enum: Stage 1 protocol selector (stored in proof)

Add to `jolt-core/src/zkvm/proof_serialization.rs` (or `jolt-core/src/zkvm/config.rs` if you prefer; must be serializable):

```rust
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpartanOuterStage1Kind {
    /// Current production Stage 1:
    /// uni-skip first round (Spartan outer) + remainder sumcheck.
    UniSkipPlusRemainder {
        remainder_impl: OuterStage1RemainderImpl,
        schedule: OuterStreamingScheduleKind,
    },

    /// Option B: full outer sumcheck protocols (no uni-skip split).
    FullBaseline,
    FullNaive,
    FullRoundBatched,
}
```

Notes:
- Keep the `UniSkipPlusRemainder { ... }` variant so you can still benchmark schedule/impl combos.
- Start with the 4 kinds above; can extend later.

### New enum: Stage 1 proof payload (stored in proof)

Add to `jolt-core/src/zkvm/proof_serialization.rs`:

```rust
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub enum Stage1Proof<F: JoltField, FS: Transcript> {
    UniSkipPlusRemainder {
        uni_skip: UniSkipFirstRoundProof<F, FS>,
        remainder: SumcheckInstanceProof<F, FS>,
    },
    FullOuter {
        sumcheck: SumcheckInstanceProof<F, FS>,
    },
}
```

### Modify `JoltProof` to use the new Stage 1 fields

In `jolt-core/src/zkvm/proof_serialization.rs`, replace:
- `stage1_uni_skip_first_round_proof`
- `stage1_sumcheck_proof`

with:

```rust
pub stage1_kind: SpartanOuterStage1Kind,
pub stage1_proof: Stage1Proof<F, FS>,
```

No backward-compat required, so it’s OK for this to break old proof deserialization.

---

## Prover changes

### Where to change

Primary file: `jolt-core/src/zkvm/prover.rs`

Stage 1 is currently produced by:

```663:779:jolt-core/src/zkvm/prover.rs
#[tracing::instrument(skip_all)]
fn prove_stage1(...) -> (UniSkipFirstRoundProof<...>, SumcheckInstanceProof<...>) { ... }
```

And then inserted into the proof in `prove()`:

```467:516:jolt-core/src/zkvm/prover.rs
let (stage1_uni_skip_first_round_proof, stage1_sumcheck_proof) = self.prove_stage1();
...
let proof = JoltProof {
    stage1_uni_skip_first_round_proof,
    stage1_sumcheck_proof,
    ...
};
```

### What to implement

1) Change `prove_stage1` to return `(SpartanOuterStage1Kind, Stage1Proof<...>)`.

2) Default kind (if no env var set):
   - `SpartanOuterStage1Kind::UniSkipPlusRemainder { remainder_impl: Streaming, schedule: LinearOnly }`
   - this mirrors the existing default behavior (see `OuterStage1Config::default()` in `jolt-core/src/zkvm/config.rs`).

3) Implement each variant:

#### A) UniSkipPlusRemainder (existing path; keep it)
- Run `OuterUniSkipProver::initialize` + `prove_uniskip_round`
- Run remainder sumcheck exactly like today (dispatch on `OuterStage1RemainderImpl` and schedule)
- Return:
  - `kind = SpartanOuterStage1Kind::UniSkipPlusRemainder { ... }`
  - `proof = Stage1Proof::UniSkipPlusRemainder { uni_skip, remainder }`

#### B) FullBaseline (new)
- Build the baseline full-outer prover instance and prove a single sumcheck instance.
- Use the already-available type:
  - Prover: `OuterBaselineSumcheckProver` in `jolt-core/src/zkvm/spartan/outer_baseline.rs`
  - It has `pub fn gen<ProofTranscript: Transcript>(..., transcript: &mut ProofTranscript) -> Self`
    and is a `SumcheckInstanceProver`.
- **Important transcript rule:** the prover must sample the same challenges the verifier will.
  - `OuterBaselineSumcheckProver::gen` already samples `tau` via `transcript.challenge_vector_optimized(...)` internally.
- Use `BatchedSumcheck::prove(vec![&mut instance], ...)` to produce a `SumcheckInstanceProof`.
- Return:
  - `kind = SpartanOuterStage1Kind::FullBaseline`
  - `proof = Stage1Proof::FullOuter { sumcheck }`

#### C) FullNaive (new)
- Use `OuterNaiveSumcheckProver` in `jolt-core/src/zkvm/spartan/outer_naive.rs` (already implements `SumcheckInstanceProver`).
- It similarly samples `tau` internally in `gen(...)`.
- Prove and return `Stage1Proof::FullOuter`.

#### D) FullRoundBatched (new)
- Use `OuterRoundBatchedSumcheckProver` in `jolt-core/src/zkvm/spartan/outer_round_batched.rs`.
- Prove and return `Stage1Proof::FullOuter`.

4) Update `prove()` to insert the new fields into `JoltProof`.

---

## Verifier changes

### Where to change

Primary file: `jolt-core/src/zkvm/verifier.rs`, function `verify_stage1`.

### What to implement

Update `verify_stage1()` to:

1) Read `self.proof.stage1_kind` and `self.proof.stage1_proof`
2) `match` on `stage1_kind` and verify the corresponding Stage 1 proof variant.

Pseudo-structure:

```rust
match (self.proof.stage1_kind, &self.proof.stage1_proof) {
  (UniSkipPlusRemainder{..}, Stage1Proof::UniSkipPlusRemainder{uni_skip, remainder}) => { ... }
  (FullBaseline, Stage1Proof::FullOuter{sumcheck}) => { ... }
  (FullNaive, Stage1Proof::FullOuter{sumcheck}) => { ... }
  (FullRoundBatched, Stage1Proof::FullOuter{sumcheck}) => { ... }
  _ => return Err(anyhow!("stage1_kind does not match stage1_proof variant"))
}
```

#### A) UniSkipPlusRemainder verification (existing)
- Run `verify_stage1_uni_skip(uni_skip, ...)` to recover `uni_skip_params`
- Instantiate `OuterRemainingSumcheckVerifier::new(...)`
- Run `BatchedSumcheck::verify(remainder, vec![&outer_remaining], ...)`

#### B) FullBaseline verification (new)
Need to verify the full outer sumcheck directly.

Verifier instance exists:
- `OuterBaselineSumcheckVerifier` in `jolt-core/src/zkvm/spartan/outer_baseline.rs`

However, it requires `tau`:

```48:61:jolt-core/src/zkvm/spartan/outer_baseline.rs
pub fn new(num_step_bits: usize, num_constraint_bits: usize, tau: Vec<F::Challenge>, key: UniformSpartanKey<F>)
```

So verifier must sample the same `tau` from the transcript as the prover did in `OuterBaselineSumcheckProver::gen`.

**How to sample `tau` cleanly:**
- Compute:
  - `num_step_bits = proof.trace_length.next_power_of_two().log_2()` (or use the same method the baseline prover used; it uses `trace.len().log_2()` on the padded trace it receives)
  - `padded_num_constraints = R1CS_CONSTRAINTS.len().next_power_of_two()`
  - `num_constraint_bits = padded_num_constraints.log_2()`
  - `total_num_vars = num_step_bits + num_constraint_bits`
- Sample:
  - `let tau: Vec<F::Challenge> = self.transcript.challenge_vector_optimized::<F>(total_num_vars);`
    (this must match the prover’s call inside `OuterBaselineSumcheckProver::gen`)

Then:
- Construct verifier: `OuterBaselineSumcheckVerifier::new(num_step_bits, num_constraint_bits, tau, self.spartan_key)`
- Verify: `BatchedSumcheck::verify(stage1_sumcheck, vec![&baseline_verifier], ...)`

#### C) FullRoundBatched verification (new)
Verifier instance exists:
- `OuterRoundBatchedSumcheckVerifier` in `jolt-core/src/zkvm/spartan/outer_round_batched.rs`
It also expects `tau` similarly. Sample `tau` with the same total variables and pass in.

#### D) FullNaive verification (missing today)
There is a prover `OuterNaiveSumcheckProver`, but no verifier type in `outer_naive.rs`.

Implement:
- `OuterNaiveSumcheckVerifier` with the *same structure as baseline verifier*:
  - It should compute `expected_output_claim` as `EqPolynomial::mle(tau, r_rev) * key.evaluate_inner_sum_product_at_point(rx_constr, r1cs_input_evals)`
  - It should `cache_openings` by appending the same witness openings at `r_cycle`.
- Then wire it in `verify_stage1`.

Rationale: baseline vs naive differ in *prover algorithm*, not in the final statement; verifier can be identical across them.

---

## Config / env var wiring for benchmarking

Add env var parsing in one place (preferably the host benchmark harness, not core).

Suggested env vars:
- `SPARTAN_OUTER_STAGE1_KIND` in `{uniskip, full-baseline, full-naive, full-round-batched}`
- If `uniskip`:
  - `OUTER_STAGE1_REMAINDER_IMPL` in `{streaming, streaming-mtable, checkpoint}` (+ later coeffmul)
  - `OUTER_STAGE1_SCHEDULE` in `{linear-only, half-split}`

Where to apply:
- `examples/sha2-chain/src/bench.rs` can print these and include them in trace filenames (similar to RA harness).
- The actual prover should be constructed with the chosen `stage1_kind` (e.g., store it in the prover struct, or pass as argument).

---

## Trace analysis / “% of total prove” measurement plan

For end-to-end benchmarking, the cleanest metric for “Spartan outer time” under Stage 1 variants is:
- **Stage 1 span time** (`prove_stage1`) as a % of `prove`.

This automatically includes:
- uni-skip + remainder (if chosen), or
- full-outer sumcheck (if chosen),
and avoids needing to enumerate internal subspans.

If you want finer breakdown later, extend `scripts/analyze_trace.py` categorization rules to include:
- `OuterBaselineSumcheckProver::compute_message`
- `OuterNaiveSumcheckProver::compute_message`
- `OuterRoundBatchedSumcheckProver::compute_message`
- `OuterRemainingSumcheckProverNonStreaming::compute_message` (uni-skip remainder variant)

---

## Implementation order checklist (for a “dumb” agent)

1) **Modify proof structs**:
   - Edit `jolt-core/src/zkvm/proof_serialization.rs`
   - Add `SpartanOuterStage1Kind` + `Stage1Proof`
   - Replace `stage1_uni_skip_first_round_proof` + `stage1_sumcheck_proof` in `JoltProof`

2) **Update prover**:
   - Edit `jolt-core/src/zkvm/prover.rs`
   - Change `prove_stage1` signature + its caller in `prove()`
   - Implement 4 variants using existing provers:
     - existing uni-skip + remainder
     - `OuterBaselineSumcheckProver`
     - `OuterNaiveSumcheckProver`
     - `OuterRoundBatchedSumcheckProver`

3) **Update verifier**:
   - Edit `jolt-core/src/zkvm/verifier.rs`
   - Rewrite `verify_stage1()` to dispatch by `stage1_kind`
   - Implement `OuterNaiveSumcheckVerifier` in `jolt-core/src/zkvm/spartan/outer_naive.rs`
   - Ensure `tau` sampling matches the prover’s sampling order for full-outer variants

4) **Benchmark harness knob**:
   - Edit `examples/sha2-chain/src/bench.rs`
   - Add env var parsing + trace name suffix including stage1 kind
   - Print the chosen stage1 kind in output

5) **Run**:
   - `cargo fmt`
   - `cargo clippy --workspace --all-targets`
   - Run `sha2-chain-bench` once per stage1 kind and ensure verification passes.

---

## Acceptance criteria

- `sha2-chain-bench` succeeds and verifies for:
  - `SPARTAN_OUTER_STAGE1_KIND=uniskip` (default)
  - `SPARTAN_OUTER_STAGE1_KIND=full-baseline`
  - `SPARTAN_OUTER_STAGE1_KIND=full-naive`
  - `SPARTAN_OUTER_STAGE1_KIND=full-round-batched`
- Trace files include the stage1 kind in their filename.
- `scripts/analyze_trace.py` reports consistent total prove time and can be used to compute Stage 1 % via the `prove_stage1` span.

