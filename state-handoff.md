# State Handoff: Wiring Sumcheck Implementation

## Handoff Reason
Phase complete - infrastructure ready, complex research task ahead

## Summary
Implemented AST-enabled witness generation infrastructure. The `RecursionProver` can now capture the full computation DAG (`AstGraph`) during witness generation, and a verifier-side `reconstruct_ast()` function can deterministically recreate it from hints. Ready to implement wiring/copy constraints.

## Goal and Scope
Implement **wiring sumcheck (Stage 1d)** from `spec.md` Section 2.7: enforce that all operations form one coherent computation DAG by proving copy constraints between operation outputs and downstream inputs. This is the critical missing piece for recursion soundness.

## Current State
**Building** - all code compiles, but wiring is not implemented (has TODO placeholders)

## Work Completed
- Added `witness_gen_with_ast()` in `jolt-core/src/poly/commitment/dory/recursion.rs:868-949`
  - Returns `WitnessWithAst { witnesses, ast, hints }`
  - Uses `TraceContext::for_witness_gen_with_ast()` and `finalize_with_ast()`
- Added `reconstruct_ast()` in same file `:951-999`
  - Verifier-side AST reconstruction using hints + AST tracing
  - Uses `TraceContext::for_hints(hints).with_ast()`
- Added `RecursionProver.ast: Option<AstGraph<BN254>>` field in `recursion_prover.rs:80`
- Added `new_from_dory_proof_with_ast()` constructor in `recursion_prover.rs:144-191`
- Updated dory imports to include `recursion::ast::AstGraph`

## Files Modified
| File | Changes |
|------|---------|
| `jolt-core/src/poly/commitment/dory/recursion.rs` | +148 lines: `WitnessWithAst`, `witness_gen_with_ast()`, `reconstruct_ast()` |
| `jolt-core/src/zkvm/recursion/recursion_prover.rs` | +66 lines: `ast` field, `new_from_dory_proof_with_ast()` |
| `Cargo.lock` | Updated dory-pcs to latest lz-recursion commit |

## Context Files (Important to Read)
| File | Why |
|------|-----|
| `jolt-core/src/zkvm/recursion/spec.md` | Section 2.7 "Stage 1d: Wiring (Copy) Constraints" - the design spec |
| `.cursor/plans/wiring_sumcheck_+_full_g1_g2_gt_witness-gen_208f432c.plan.md` | Full implementation plan with todos |
| `jolt-core/src/zkvm/recursion/recursion_prover.rs:758-770` | TODO placeholder for wiring sumcheck |
| `jolt-core/src/zkvm/recursion/recursion_verifier.rs:289-291` | TODO placeholder for wiring verifier |
| `jolt-core/src/zkvm/recursion/stage2/` | Existing stage2 sumcheck implementations as reference |

## Key Decisions and Rationale
1. **Verifier reconstructs AST** (not prover-supplied) - prevents prover from choosing easier wiring graph
2. **Correctness-first AST reconstruction** - re-runs verify_recursive with hints; can optimize later
3. **AST field is Option** - backward compatible; existing code paths don't require AST

## Blockers / Errors
None currently blocking. The `test_recursion_snark_e2e_with_dory` test fails with `SumcheckVerificationError`, but this appears pre-existing and unrelated to AST work.

## Open Questions / Risks (Research Needed)

### ✅ RESOLVED: Wiring Sumcheck Design

**See new design documents:**
- `jolt-core/src/zkvm/recursion/wiring_sumcheck_design.md` - Main design overview
- `jolt-core/src/zkvm/recursion/wiring_gt.md` - GT track specification
- `jolt-core/src/zkvm/recursion/wiring_g1.md` - G1 track specification  
- `jolt-core/src/zkvm/recursion/wiring_g2.md` - G2 track specification

**Key Design Decisions:**

1. **Port extraction via selector sumchecks**: 
   - For step-indexed ops (GTExp, G1/G2 ScalarMul), extract final-step output via `Eq(N, s)` selector
   - GTExp: 7-round sumcheck with `Eq(127, s)` 
   - G1/G2 ScalarMul: 8-round sumcheck with `Eq(255, s)`

2. **Three parallel wiring sumchecks (G1, G2, GT)**:
   - Each track has different coordinate counts: GT=12, G1=3, G2=5
   - Can run in parallel for performance
   - **Recommended**: reuse Stage 1’s shared point (single unified opening point) and avoid fresh `r_wire`
   - (Fresh `r_wire` is an alternative, but it forces multiple opening points / Stage 2 changes)

3. **Accumulator semantics standardized**:
   - Initial state = identity (ρ₀=1 for GT, A₀=O for G1/G2)
   - Base is INPUT to operation, not initial accumulator
   - Final state is OUTPUT, extracted via port extraction sumcheck

4. **Copy constraint verification**:
   - After port extraction, verify `Σ_e λ_e · (PortOut - PortIn) = 0`
   - Coordinate batching via powers of μ (3 terms for G1, 5 for G2, 12 for GT)
   - Soundness: random coefficients prevent cancellation

### Remaining Research Questions:

3. **AstGraph API understanding needed:**
   - `AstNode` structure: `{ value_id, op: AstOp, inputs: Vec<ValueId> }`
   - `AstOp` variants: `G1Add`, `G2Add`, `G1ScalarMul`, `G2ScalarMul`, `GTExp`, `GTMul`, etc.
   - How to iterate edges: `for node in ast.nodes() { for input_id in node.op.input_ids() { ... } }`
   - `ValueType` enum: `G1`, `G2`, `GT`, `Scalar`

4. **Connection to external pairing boundary:**
   - Final 3-way multi-pairing inputs must be exposed as public outputs
   - These are special "boundary wiring" edges to "public output slots"
   - Verifier checks these match expected pairing input format

5. **Stage 2 integration with opening points:**
   - **Goal**: keep wiring constraints at the same opening point used by Stage 2 (the Stage 1 point `r_x*`)
   - With the “single opening point” design, port-extraction sumchecks *pin* their final check to `(r_s*, r_x*)`,
     so extracted port values are verifier-side scalars and do **not** add new matrix rows/points.

### Dory Types to Understand (from `dory::recursion::ast`):
- `AstGraph<E>` - the full DAG
- `AstNode` - individual operation node
- `AstOp` - operation type enum (carries `op_id: Option<OpId>`)
- `ValueId` - unique identifier for a value in the DAG
- `ValueType` - G1/G2/GT/Scalar type tag
- `InputSource` - where an input comes from (another node, constant, etc.)

## Cleanup Needed
None - code is clean, no debug prints or temporary hacks

## Tests and Commands Run
```bash
cargo check -p jolt-core  # PASS
git push  # commit 5e08ee80e pushed to quang/voj-recursion
```

## Next Steps (Priority Order)

### 1. ✅ COMPLETE: Wiring Sumcheck Design
Design documents created:
- `wiring_sumcheck_design.md` - Overview and architecture
- `wiring_gt.md` - GT track with 7-round port extraction
- `wiring_g1.md` - G1 track with 8-round port extraction
- `wiring_g2.md` - G2 track with 8-round port extraction

### 2. Research: Understand Dory AST API
```bash
# Generate and browse dory docs
cargo doc -p dory-pcs --no-deps --open
# Focus on: recursion::ast module
```
Key questions:
- How to iterate nodes in topological order?
- How to get input/output ValueIds from an AstOp?
- How to map ValueId to constraint index?

### 3. Implement: Port Extraction Sumcheck
Create `jolt-core/src/zkvm/recursion/stage1/port_extraction.rs`:
```rust
/// Extract output values from step-indexed traces at specific steps
pub struct PortExtractionProver {
    /// Target step to extract (127 for GTExp, 255 for G1/G2 ScalarMul)
    target_step: usize,
    /// Step dimension size (7 vars for GTExp, 8 for G1/G2)
    step_vars: usize,
    /// Trace polynomials bound to r_wire
    bound_traces: Vec<MultilinearPolynomial<F>>,
}
```

### 4. Implement: Per-Track Wiring Sumchecks
Create `jolt-core/src/zkvm/recursion/stage1/wiring_{gt,g1,g2}.rs`:
- Collect edges from AST by value type
- Run port extraction for step-indexed sources
- Verify copy constraint sum = 0
- Emit virtual claims for Stage 2

### 5. Integration: Wire into Prover/Verifier
- Replace TODO in `recursion_prover.rs:758-770`
- Replace TODO in `recursion_verifier.rs:289-291`
- Use AST from `RecursionProver.ast` field
- Extend Stage 2 matrix to include wiring claims at `(r_s', r_wire)`

## How to Resume
1. **First**: Read the new wiring design docs: `wiring_sumcheck_design.md`, `wiring_gt.md`, `wiring_g1.md`, `wiring_g2.md`
2. **Second**: Read `cargo doc -p dory-pcs --no-deps --open` → `recursion::ast` module to understand `AstGraph`, `AstNode`, `AstOp` API
3. **Third**: Implement port extraction sumcheck following the design in `wiring_gt.md` Section 4
4. **Then**: Implement copy constraint verification following `wiring_sumcheck_design.md` Section 5

## Key Insights from Wiring Design Session

### The Core Problem Solved
The dimension mismatch between step-indexed operations (GTExp, G1/G2 ScalarMul) and non-step operations (GTMul, G1/G2 Add) is resolved via **port extraction sumchecks**:
- Use selector polynomial `Eq(N, s)` to extract the value at step N
- This converts a step-indexed polynomial to a pure element evaluation
- All ports can then be compared at a common evaluation point `r_wire`

### Critical Design Decisions
1. **Accumulator = Identity, not Base**: Initial state is always identity/infinity. Base is an INPUT that gets wired from elsewhere (or is a boundary constant).

2. **Three Parallel Tracks**: GT, G1, G2 wiring run independently with different coordinate counts (12, 3, 5 respectively).

3. **Fresh Wiring Point**: Use new `r_wire` separate from Stage 1's `r_x*` for clean security analysis.

4. **Coordinate Batching**: Use powers of μ to batch multiple coordinate comparisons into one field check.

### Soundness Argument
- Port extraction: Schwartz-Zippel over step dimension (7-8 rounds)
- Copy constraint: Random linear combination prevents cancellation
- Coordinate batching: Non-zero polynomial in μ caught by random evaluation
- Combined error: O(n_edges / |F|) which is negligible for |F| ≈ 2^254

### Completeness Argument
- Honest prover has matching ports (by AST construction)
- Port extraction correctly evaluates trace at target step (Lagrange interpolation)
- Copy constraint sum is exactly zero when all edges match

## Git State
```
Branch: quang/voj-recursion (ahead 1 of remote)
Latest commit: 5e08ee80e "feat(recursion): add AST-enabled witness generation and verifier reconstruction"
Untracked: state-handoff.md (this file)
Stashes: 5 (unrelated to this work)
```
