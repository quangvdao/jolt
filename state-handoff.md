## State handoff (2026-01-16)

## Update (2026-01-17)

The rest of this document is an **older handoff** and contains stale details (e.g. trace-oracle / host patch hooks).
The current workspace state is:

- **Provable-only Grumpkin MSM crate**: `jolt-inlines/grumpkin_msm` (no trace-oracle / host mode).
- **Hyrax integration**: `HyraxOpeningProof::verify` will use `jolt_inlines_grumpkin_msm::grumpkin_msm_2048`
  when `jolt-core` is built with feature `grumpkin-msm-provable` (see `jolt-core/src/poly/commitment/hyrax.rs`).
- **Guest fallback MSM correctness**: Hyrax’s non-accelerated guest MSM path now uses Arkworks serial MSM
  (`VariableBaseMSM::msm_field_elements`) instead of an ad-hoc Pippenger.
- **Stage 8 verification**: currently always uses the non-hint path to keep transcript evolution stable across
  environments (see `jolt-core/src/zkvm/verifier.rs`).

### Commands (repro)

1. Regenerate the proof bundle (required if serialization changes):
   - `CARGO_NET_OFFLINE=true RUST_LOG=info cargo run --offline --release -p recursion -- generate --example fibonacci --workdir output`
2. Baseline trace (no Grumpkin MSM feature):
   - `CARGO_NET_OFFLINE=true RUST_LOG=info cargo run --offline --release -p recursion -- trace --example fibonacci --workdir output --embed --disk`
3. Trace with provable Grumpkin MSM enabled in the `recursion-guest` build:
   - `CARGO_NET_OFFLINE=true RUST_LOG=info JOLT_GUEST_EXTRA_FEATURES_PKG=recursion-guest JOLT_GUEST_EXTRA_FEATURES=grumpkin-msm-provable cargo run --offline --release -p recursion -- trace --example fibonacci --workdir output --embed --disk`

### Latest cycle result (summary)

From the reruns above (both return recursion output `1` = verification succeeded):
- Baseline `hyrax_verify_total`: **517,898,328** virtual cycles
- With `grumpkin-msm-provable` `hyrax_verify_total`: **529,469,592** virtual cycles

- **Handoff reason:** phase complete (Grumpkin curve-op inline scaffolding + tests; provable field/curve ops not implemented yet)

- **Summary:** Extended the tracer to support **trace-time patch hooks** for `.insn` inlines, then added a new `jolt-inlines/grumpkin_msm` crate that currently implements **trace-only** Grumpkin MSM(2048) plus **trace-only curve ops** (Jacobian double + mixed add) using stable `#[repr(C)]` point/field layouts. Hyrax verifier is feature-gated to optionally use the MSM inline in the RISC-V guest, falling back to a serial MSM to avoid Rayon panics.

- **Goal and scope:** Implement a **proper (provable + fast)** Grumpkin MSM inside Hyrax verification (recursion verifier hot path). The intended architecture is “MSM loops/branches in guest Rust + straight-line provable `.insn` primitives (Fq ops, curve add/double)” as outlined in `grumpkin-msm-provable-plan.md`.

- **Current state:** building (host) + partial tests passing
  - `cargo check -p tracer -p jolt-core -p recursion` PASS (one warning in recursion example; see below)
  - `cargo test -p jolt-inlines-grumpkin-msm --features host` PASS (7 tests)

- **Work completed:**
  - **Tracer patch-hook support**:
    - Added optional trace patch hook type + stored in registry: `tracer/src/instruction/inline.rs:L25-L42`
    - Extended `register_inline(..., trace_patch_fn)` API: `tracer/src/instruction/inline.rs:L57-L64`
    - Apply patch hook before tracing inline sequence: `tracer/src/instruction/inline.rs:L203-L216`
    - Important constraint (design impact): inline sequence executes as a straight-line list: `tracer/src/instruction/inline.rs:L235-L237`
  - **Updated existing inlines to new `register_inline` API** (pass `None` patch hook):
    - SHA2: `jolt-inlines/sha2/src/host.rs:L14-L31`
    - Keccak256: `jolt-inlines/keccak256/src/host.rs:L11-L19`
    - Blake2: `jolt-inlines/blake2/src/host.rs:L9-L17`
    - Blake3: `jolt-inlines/blake3/src/host.rs:L12-L29`
    - BigInt mul: `jolt-inlines/bigint/src/lib.rs:L16-L27`
  - **Hyrax verifier MSM refactor + feature-gated MSM inline path**:
    - `try_grumpkin_msm2048` closure (Grumpkin-only via `Any` type checks) + call sites: `jolt-core/src/poly/commitment/hyrax.rs:L304-L395`
    - Cycle spans around hot MSMs: `jolt-core/src/poly/commitment/hyrax.rs:L273-L370`
  - **Serial MSM helper to avoid Rayon-in-guest issues**:
    - `msm_field_elements` uses arkworks `msm_serial`: `jolt-core/src/msm/mod.rs:L120-L127`
    - Host test for correctness vs naive Grumpkin MSM: `jolt-core/src/msm/mod.rs:L226-L256`
  - **Guest build plumbing for opt-in profiling features**:
    - Extra guest features via env vars: `jolt-core/src/host/program.rs:L195-L224`
    - Target triple selection (std vs no-std): `jolt-core/src/host/program.rs:L141-L145`
  - **Feature plumbing (workspace → sdk → recursion guest)**:
    - Workspace member: `Cargo.toml:L25-L39` (includes `jolt-inlines/grumpkin_msm`)
    - Workspace dep: `Cargo.toml:L240-L252` (`jolt-inlines-grumpkin-msm`)
    - `jolt-core` feature + deps: `jolt-core/Cargo.toml:L25-L38`, `jolt-core/Cargo.toml:L51`, `jolt-core/Cargo.toml:L110-L111`
    - `jolt-sdk` forwards feature: `jolt-sdk/Cargo.toml:L42`
    - `recursion-guest` forwards feature: `examples/recursion/guest/Cargo.toml:L6-L9`
  - **Force-link inline crate so `#[ctor]` registration runs**:
    - `use jolt_inlines_grumpkin_msm as _;`: `jolt-core/src/host/mod.rs:L3-L7`
  - **CycleSpan helper (guest cycle markers)**:
    - RAII helper: `jolt-core/src/utils/cycle_span.rs:L1-L23`
    - Module export: `jolt-core/src/utils/mod.rs:L7`
  - **New inline crate: `jolt-inlines/grumpkin_msm` (currently trace-only/oracle semantics)**:
    - Safety warning / current limitations: `jolt-inlines/grumpkin_msm/src/lib.rs:L3-L8`
    - Inline IDs (opcode/funct): `jolt-inlines/grumpkin_msm/src/lib.rs:L12-L27`
    - Stable `#[repr(C)]` limb layouts + host conversions: `jolt-inlines/grumpkin_msm/src/types.rs:L6-L191`
    - Scalar bit/window helpers (for future MSM digit extraction): `jolt-inlines/grumpkin_msm/src/types.rs:L19-L58`
    - MSM(2048) guest `.insn` wrapper: `jolt-inlines/grumpkin_msm/src/sdk.rs:L6-L37`
    - Curve-op guest `.insn` wrappers:
      - Jacobian double: `jolt-inlines/grumpkin_msm/src/sdk.rs:L39-L61`
      - Mixed add: `jolt-inlines/grumpkin_msm/src/sdk.rs:L63-L85`
    - Inline sequences (all currently “advice→store” placeholders):
      - MSM: `jolt-inlines/grumpkin_msm/src/sequence_builder.rs:L32-L52`
      - Jacobian double: `jolt-inlines/grumpkin_msm/src/sequence_builder.rs:L54-L67`
      - Mixed add: `jolt-inlines/grumpkin_msm/src/sequence_builder.rs:L69-L82`
    - Host registration + trace patch hooks:
      - Registration: `jolt-inlines/grumpkin_msm/src/host.rs:L18-L48`
      - MSM oracle patcher (reads arkworks layouts from guest memory): `jolt-inlines/grumpkin_msm/src/host.rs:L50-L115`
      - Curve-op oracle patchers (use stable `types::*` layout): `jolt-inlines/grumpkin_msm/src/host.rs:L117-L175`
    - Curve-op inline tests (host-only, tracer harness): `jolt-inlines/grumpkin_msm/src/curve_ops_tests.rs:L69-L173`
    - Crate features/dev-deps: `jolt-inlines/grumpkin_msm/Cargo.toml:L10-L35`
  - **Roadmap doc (provable MSM plan)**:
    - Key constraint + design direction: `grumpkin-msm-provable-plan.md:L11-L23`
    - Phase 0 feature split (oracle vs provable): `grumpkin-msm-provable-plan.md:L52-L61`

- **Files modified (tracked):**
  - `tracer/src/instruction/inline.rs`: trace patch hook plumbing
  - `jolt-core/src/poly/commitment/hyrax.rs`: Hyrax verify MSM refactor + feature-gated inline call
  - `jolt-core/src/msm/mod.rs`: `msm_field_elements` (serial) + test
  - `jolt-core/src/host/mod.rs`: force-link inline crate for `#[ctor]` registration
  - `jolt-core/src/host/program.rs`: env-var guest feature injection
  - `jolt-core/src/utils/mod.rs`: exports `cycle_span`
  - `jolt-core/Cargo.toml`: feature wiring for `grumpkin-msm-inline` + host deps
  - `Cargo.toml`: workspace member + workspace dep; local dory override
  - Plus other modified files from prior profiling work (see `git status -sb` output below)

- **Files added / untracked (need review + likely commit):**
  - `jolt-inlines/grumpkin_msm/**` (new inline crate; includes `types.rs` + curve ops + tests)
  - `grumpkin-msm-provable-plan.md` (design doc)
  - `jolt-core/src/utils/cycle_span.rs` (cycle span helper)
  - `cycle-counts.md` (notes)
  - `third-party/dory/` (local override used by root `Cargo.toml`; decide whether to keep)
  - `RUST_BACKTRACE=1` (stray file; delete)

- **Context files (important, not modified):**
  - Inline ABI semantics: `tracer/src/instruction/format/format_inline.rs:L1-L5` (rs3 is output pointer; FormatInline doesn’t write registers)
  - Tracer inline test harness: `tracer/src/utils/inline_test_harness.rs:L21-L64` (memory layout + rs1/rs2/rs3 mapping)

- **Key decisions and rationale:**
  - **Trace patch hooks** are used to quickly validate `.insn` plumbing + measure cycles without having provable constraints yet (`VirtualAdvice` patching). See patch hook execution order: `tracer/src/instruction/inline.rs:L203-L216`.
  - **Inline straight-line constraint** means we should not attempt “single `.insn` does full MSM with branches”; instead use `.insn` for primitives and keep MSM loops in Rust: `tracer/src/instruction/inline.rs:L235-L237` and plan rationale: `grumpkin-msm-provable-plan.md:L11-L23`.
  - **Stable `#[repr(C)]` layouts** (`types.rs`) are the foundation for a provable MSM path, avoiding reliance on arkworks struct layout in guest memory: `jolt-inlines/grumpkin_msm/src/types.rs:L1-L112`.
  - **Serial MSM in guest** avoids Rayon threadpool creation panics (guest env limitation). See `msm_field_elements`: `jolt-core/src/msm/mod.rs:L120-L127`.

- **Blockers / Errors:** none currently.
  - Note: `cargo check -p tracer -p jolt-core -p recursion` emits a warning:
    - `examples/recursion/src/main.rs:391:9`: unused variable `verifier_preprocessing` (non-fatal).

- **Open questions / Risks:**
  - **Soundness risk:** MSM + curve ops are currently **trace-only oracles** (they patch advice) and are NOT provable. Crate-level warning: `jolt-inlines/grumpkin_msm/src/lib.rs:L3-L8`.
  - **Layout mismatch risk (MSM only):** MSM oracle patcher still reads arkworks layouts from guest memory: `jolt-inlines/grumpkin_msm/src/host.rs:L66-L80`. Curve ops already use stable layouts.
  - **Repo hygiene risk:** `third-party/dory/` is untracked but referenced by root override `Cargo.toml:L151-L155`; decide whether to commit it or revert the override.
  - **Artifact risk:** `output/fibonacci-guest_proofs.bin` is tracked and changed; confirm whether it should be committed.

- **Cleanup needed:**
  - Delete `RUST_BACKTRACE=1` (untracked).
  - Decide what to do with `third-party/dory/` + `Cargo.toml` override (`Cargo.toml:L151-L155`).
  - Consider fixing the recursion example warning (`examples/recursion/src/main.rs:391`).

- **Tests and commands run:**
  - `git status -sb` (2026-01-16): `quang/grumpkin-msm-inline...upstream/quang/grumpkin-msm-inline`
  - `git diff --stat` (2026-01-16): 24 files changed, 607 insertions(+), 377 deletions(-)
  - `git stash list` (2026-01-16):
    - `stash@{0}: WIP on quang/dory-layout-enum: ed7e49f9 Fix bugs`
    - `stash@{1}: On quang/fix-secp256k1-inline: non-secp256k1 changes`
    - `stash@{2}: WIP on quang/prover-backend-new-ram: bd69d549 Merge quang/feat-col-major-sparse-matrix into quang/prover-backend-new-ram`
    - `stash@{3}: WIP on oakland-benchmarks: 1566973f add deg2 sumcheck bench`
  - `cargo test -p jolt-inlines-grumpkin-msm --features host` PASS
  - `cargo check -p tracer -p jolt-core -p recursion` PASS (warning noted above)

- **Next steps (priority order):**
  1. **Phase 0 safety:** split feature flags into `grumpkin-msm-trace-oracle` vs `grumpkin-msm-provable` (`grumpkin-msm-provable-plan.md:L52-L61`) and ensure oracle features can’t be enabled for proof generation.
  2. **Make MSM inputs use stable layouts:** change Hyrax to convert bases/scalars into `types::{AffinePoint, FrLimbs}` and update MSM patcher to read those instead of arkworks layouts (`jolt-inlines/grumpkin_msm/src/host.rs:L66-L80`).
  3. **Implement provable primitives:** implement Fq arithmetic inlines, then rewrite curve-op sequence builders to compute (no `VirtualAdvice`) and remove patch hooks for provable builds (see plan starting at `grumpkin-msm-provable-plan.md:L85`).
  4. **MSM proper:** implement a fixed-window/pippenger MSM loop in guest Rust using the provable curve ops + precomputed scalar digits (`types::FrLimbs::window`: `jolt-inlines/grumpkin_msm/src/types.rs:L36-L57`).
  5. **Cycle measurement:** run recursion trace with inline/provable feature enabled and compare spans `hyrax_verify_msm_rows` / `hyrax_verify_msm_product` (`jolt-core/src/poly/commitment/hyrax.rs:L338-L395`).

- **How to resume:**
  1. `cargo test -p jolt-inlines-grumpkin-msm`
  2. Guest trace run (for cycle profiling):
     - `CARGO_NET_OFFLINE=true RUST_LOG=info cargo run --offline --release -p recursion -- trace --example fibonacci --workdir output --embed --disk`
     - `CARGO_NET_OFFLINE=true RUST_LOG=info JOLT_GUEST_EXTRA_FEATURES_PKG=recursion-guest JOLT_GUEST_EXTRA_FEATURES=grumpkin-msm-provable cargo run --offline --release -p recursion -- trace --example fibonacci --workdir output --embed --disk`

