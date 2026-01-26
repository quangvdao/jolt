# jolt-recursion

Recursion prover/verifier for Jolt, built on top of `jolt-core`'s public stage verification APIs.

## Overview

This crate provides an **optional, modular** recursion layer for Jolt proofs. The base `jolt-core` proof is completely recursion-free; recursion is a separate artifact that wraps around a base proof.

The recursion SNARK proves that the Dory polynomial commitment scheme's Stage 8 verification (the expensive pairing-based opening check) was performed correctly, reducing it to a single external pairing check.

## Key Features

- **Modular**: Recursion is opt-in. Base proofs work without it; add recursion only when needed.
- **Separate artifacts**: Base proof and recursion proof are independent, serializable objects.
- **Composable**: Generate base proof first, then layer recursion on top (or skip it entirely).
- **Clean separation**: `jolt-core` has zero recursion dependencies; all recursion logic lives here.

## Architecture

### Proving Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   1. Base Jolt Prover                       │
│                      (jolt-core)                            │
│  - Execute guest program                                    │
│  - Generate witness polynomials                             │
│  - Commit with Dory PCS                                     │
│  - Run sumcheck protocols (Stages 1-7)                      │
│  - Generate Stage 8 opening proof                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      JoltProof (base)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                2. Recursion Prover                          │
│                   (jolt-recursion)                          │
│  - Replay Stages 1-7 to reconstruct transcript              │
│  - Generate witness for Dory verification circuit           │
│  - Prove G1/G2/GT constraint systems via sumcheck           │
│  - Commit dense witness with Hyrax                          │
│  - Output pairing boundary points                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     RecursionProof
```

### Verification Flow

```
┌─────────────────────────────────────────────────────────────┐
│              1. Base Verification (Stages 1-7)              │
│  - Verify sumcheck proofs                                   │
│  - Check constraint satisfiability                          │
│  - Reconstruct transcript state                             │
│  ─────────────────────────────────────────────────────────  │
│  Stage 8 (Dory PCS) is SKIPPED - handled by recursion       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              2. Recursion SNARK Verification                │
│  - Verify recursion sumcheck proofs                         │
│  - Verify Hyrax opening proof                               │
│  - Extract pairing boundary points                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              3. External Pairing Check                      │
│  - Single multi-pairing (3 pairings):                       │
│    e(P1,Q1) · e(P2,Q2) · e(P3,Q3) = RHS                     │
│  - This is the ONLY pairing operation needed                │
└─────────────────────────────────────────────────────────────┘
```

### Modularity in Practice

```
Without recursion:          With recursion:
                            
  Base Prover                 Base Prover
      │                           │
      ▼                           ▼
  JoltProof ──────────────►  JoltProof
      │                           │
      ▼                           ▼
  Base Verifier              Recursion Prover
  (full Stage 8)                  │
      │                           ▼
      ▼                      RecursionProof
    Done                          │
                                  ▼
                             Recursion Verifier
                             (cheap pairing)
                                  │
                                  ▼
                                Done
```

### Key Components

- **`RecursionProof`**: The recursion proof artifact containing:
  - Sumcheck proofs for recursion constraints
  - Hyrax opening proof for dense witness polynomial
  - Pairing boundary points for external verification

- **`RecursionExt` trait**: Extension trait for commitment schemes that support recursion. Currently implemented for Dory.

- **`prove_recursion`**: Generates a recursion proof given a base Jolt proof
- **`verify_recursion`**: Verifies a recursion proof against a base Jolt proof

## Quick Start

### Running Tests

```bash
# Run all jolt-recursion tests
cargo test -p jolt-recursion

# Run a specific test
cargo test -p jolt-recursion recursion_proof_roundtrip
```

### Example: Recursion with the CLI

The `examples/recursion` binary demonstrates recursion proving with the `--recursion` flag:

```bash
# Generate proofs for fibonacci WITH recursion
cargo run -p recursion --release -- generate \
    --example fibonacci \
    --workdir output \
    --recursion

# Generate proofs for fibonacci WITHOUT recursion (base proof only)
cargo run -p recursion --release -- generate \
    --example fibonacci \
    --workdir output

# With committed bytecode mode + address-major layout
cargo run -p recursion --release -- generate \
    --example fibonacci \
    --workdir output \
    --committed \
    --layout address-major \
    --recursion
```

### Profiling with jolt-core CLI

```bash
# Profile with recursion enabled (note: recursion benchmarking in jolt-recursion)
cargo run -p jolt-core --release -- profile \
    --name fibonacci \
    --scale 20 \
    --recursion
```

## Programmatic Usage

```rust
use jolt_recursion::{prove_recursion, verify_recursion, RecursionProof};
use jolt_sdk::{FS, JoltVerifierPreprocessing, RV64IMACProof};

// After generating a base Jolt proof...
let base_proof: RV64IMACProof = /* ... */;
let preprocessing: JoltVerifierPreprocessing<_, _> = /* ... */;
let io_device: JoltDevice = /* ... */;

// Generate recursion proof
let recursion_proof: RecursionProof<FS> = prove_recursion::<FS>(
    &preprocessing,
    io_device.clone(),
    None, // trusted_advice_commitment
    &base_proof,
).expect("recursion proving failed");

// Verify recursion proof
verify_recursion::<FS>(
    &preprocessing,
    io_device,
    None, // trusted_advice_commitment
    &base_proof,
    &recursion_proof,
).expect("recursion verification failed");
```

## Verification Flow

1. **Stages 1-7**: Run natively (same as base Jolt verification)
2. **Stage 8 Reconstruction**: Rebuild transcript state and derive Dory verification claims
3. **Recursion SNARK Verification**: Verify sumcheck proofs for G1/G2/GT constraint systems
4. **External Pairing Check**: Verify the final pairing equation using the boundary points

## Features

- `host` (default): Enables host-side proving functionality
- `allocative`: Enables memory profiling support
- `experimental-pairing-recursion`: Experimental features for pairing-based recursion

## Crate Structure

```
jolt-recursion/
├── src/
│   └── lib.rs              # Main entry point, prove_recursion/verify_recursion APIs
├── tests/
│   ├── dory_combine_witness.rs    # Dory commitment combination tests
│   └── recursion_roundtrip.rs     # End-to-end recursion test
└── Cargo.toml
```

The recursion constraint system implementation lives in `jolt-core/src/zkvm/recursion/` and is included via `#[path]` directives to maintain `crate::` path compatibility from the original code location.

## Design Rationale

- **Separation from jolt-core**: Keeps the base proof format recursion-free and reduces compilation complexity for users who don't need recursion
- **Public stage APIs**: `jolt-core` exposes `verify_stage1..8` methods so recursion can reconstruct transcript state externally
- **Path-based includes**: Recursion code remains in `jolt-core/src/zkvm/recursion/` but is compiled only via this crate, avoiding large file moves while preserving internal paths
