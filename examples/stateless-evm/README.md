# Stateless EVM Block Validation in Jolt

This example proves (or traces) canonical **stateless** Ethereum block
validation inside the Jolt zkVM. It is the "reth + revm + MPT-verified
execution witnesses" stack that every production Ethereum zkVM today
uses, compiled as a Jolt guest program and wired up to the Jolt SDK.

## What it actually does

The guest receives a `StatelessInput` (block, execution witness, chain
config) plus pre-recovered uncompressed secp256k1 public keys for each
transaction, then calls
[`stateless::stateless_validation_with_trie`](https://github.com/paradigmxyz/stateless).
This performs the full canonical block validation pipeline:

1. Decode the execution witness into a sparse MPT of pre-state accounts
   and storage, verified against the parent block's `stateRoot`.
2. Execute the block with revm under reth's chain rules
   (`reth-evm-ethereum` + `reth-chainspec`).
3. Run consensus checks: gas limit, base fee, receipts root, logs
   bloom, requests root, withdrawals.
4. Recompute the post-state root via MPT operations and compare against
   the block header.

Returning `success: true` means all four stages passed. A dishonest
prover cannot substitute state, forge signatures, skip consensus checks,
or lie about the post-state root.

Keccak256 and secp256k1 signature verification inside the guest are
routed through Jolt inlines (`jolt-inlines-keccak256`,
`jolt-inlines-secp256k1`), which is why these proofs are tractable.
This branch also carries a local `revm-interpreter` patch that rewrites
the hottest `U256` stack and memory moves as explicit 64-bit limb
operations, rather than byte-slice copies.

## Layout

- `guest/` — `#[jolt::provable]` function `stateless_validate` plus a
  `CryptoProvider` that uses Jolt's secp256k1 inline for ECDSA
  verification and a `native_keccak256` hook for
  `alloy-primitives`.
- `host/` — CLI that loads fixtures, pre-recovers signers, compiles
  the guest, and either traces (`--mode analyze`) or prove+verifies
  (`--mode prove`) a block.
- `patches/revm-interpreter/` — local crates.io override for
  `revm-interpreter`, used to optimize `Stack::{dup,exchange}` and the
  `mstore` / `SharedMemory` write path for Jolt's 8-byte memory model.
- `fixtures/` — committed JSON fixtures:
  - `empty_block_osaka_1M.json` and `ether_transfers_osaka_1M.json`
    from the Ethereum Foundation's `zkevm-benchmark-workload` (v0.0.7).
  - `mainnet_block_22974576.json` — a real Ethereum mainnet block
    (115 txs, 7.6M gas used, 2025-07-22) copied from
    [`eth-act/ere-guests`](https://github.com/eth-act/ere-guests) at
    commit `ec5feba` (`crates/integration-tests/fixtures/block.tar.gz`),
    dual-licensed MIT OR Apache-2.0.

## Quickstart

From the repo root:

```bash
# Trace the empty-block fixture (fast: ~1s after guest compile)
cargo run --release -p stateless-evm -- \
    examples/stateless-evm/fixtures/empty_block_osaka_1M.json \
    --mode analyze

# Trace the 47-tx ether-transfer fixture (exercises secp256k1 recovery)
cargo run --release -p stateless-evm -- \
    examples/stateless-evm/fixtures/ether_transfers_osaka_1M.json \
    --mode analyze

# Trace a real Ethereum mainnet block (~464M cycles, ~65-75s on a laptop)
cargo run --release -p stateless-evm -- \
    examples/stateless-evm/fixtures/mainnet_block_22974576.json \
    --mode analyze

# Full prove + verify on a fixture
cargo run --release -p stateless-evm -- \
    examples/stateless-evm/fixtures/empty_block_osaka_1M.json \
    --mode prove
```

Useful flags (`--help` for the full list):

- `--mode {analyze,prove}` (default `analyze`).
- Input can be a directory of fixtures. `analyze` mode traces each and
  prints a batch summary. `--filter` and `--limit` narrow the batch,
  `--keep-going` continues after per-file failures.
- `--target-dir` sets the Jolt guest artifact cache
  (default `/tmp/jolt-guest-targets`).
- `--write-prepared PATH` dumps the postcard-serialized guest input
  (handy for offline debugging).

## Expected output

Analyzing `empty_block_osaka_1M.json` on a recent laptop:

```
guest compile time: ~80s   (first run only; cached after)
loaded input: name=... block_hash=0xe4bd... tx_count=0 expected_success=Some(true)
analysis complete: cycles=1862817 padded_cycles=2097152 success=true
```

`ether_transfers_osaka_1M.json` adds 47 transactions and lands around
18.2M cycles with the current memops configuration.
`mainnet_block_22974576.json` is a full-size mainnet
block (115 txs, 7.6M gas used) and lands at ~464M cycles / 2^29 padded.

## Fixture format

Each fixture JSON has the shape

```json
{
  "name": "test_transaction_types.py::test_empty_block[...]",
  "stateless_input": { "block": ..., "witness": ..., "chain_config": ... },
  "success": true
}
```

This is the exact format produced by
[`eth-act/zkevm-benchmark-workload`](https://github.com/eth-act/zkevm-benchmark-workload)'s
`witness-generator-cli` and consumed by every `stateless-validator`
zkVM host. The host binary accepts that JSON directly; see
`host/src/main.rs:56` (`StatelessValidatorFixture`). Raw
`StatelessInput` JSON (without the `name`/`success` envelope) is also
accepted, as is a postcard-serialized `PreparedStatelessInput`.

## Generating more fixtures

The committed fixtures are deliberately tiny. To generate larger or
different ones from the Ethereum Foundation's benchmark suite:

```bash
# One-time: download the EEST benchmark fixtures (~430 MB)
curl -fL -o /tmp/eest.tar.gz \
    "https://github.com/ethereum/execution-spec-tests/releases/download/benchmark%40v0.0.7/fixtures_benchmark.tar.gz"
mkdir -p ~/.cache/eest-fixtures && tar -xzf /tmp/eest.tar.gz -C ~/.cache/eest-fixtures

# Clone the benchmark workload runner
git clone https://github.com/eth-act/zkevm-benchmark-workload ~/zkevm-benchmark-workload

# Generate a filtered set of fixtures (AND semantics across --include)
cd ~/zkevm-benchmark-workload
RUST_MIN_STACK=67108864 cargo run -p witness-generator-cli --release -- \
    -o /tmp/jolt-fixtures \
    tests \
    --eest-fixtures-path ~/.cache/eest-fixtures \
    --include test_empty_block \
    --include 10M
```

The filter uses substring-contains, AND'ed across every `--include`.
Useful test names live under
`fixtures/blockchain_tests/benchmark/...` inside the tarball; grep
there to pick a test and gas-value combination that fits your
experiment.

## RPC-based fixtures

Generating fresh mainnet fixtures end-to-end requires either a Reth
archival node that exposes `debug_executionWitness`, or a proxy such
as [`zeth-rpc-proxy`](https://github.com/boundless-xyz/zeth) pointed
at an RPC that supports archive `eth_getProof` / `eth_getCode` /
`eth_getStorageAt` (Alchemy / Infura / QuickNode / a self-hosted
Reth node). Free public RPCs (drpc, publicnode, llamarpc, flashbots,
cloudflare, ...) do not expose `debug_executionWitness` and are
unreliable partway through a local block replay. The typical flow
with an archive-capable upstream URL is:

```bash
cd ~/zkevm-benchmark-workload
cargo run -p witness-generator-cli --release -- \
    -o /tmp/jolt-rpc-fixtures \
    rpc --block 20000000 --rpc-url $RPC_URL
```

The output JSON is the same `StatelessValidatorFixture` format as our
committed fixtures. For quick experimentation without any RPC access,
the simplest option is to copy another block out of
`eth-act/ere-guests`' `crates/integration-tests/fixtures/block.tar.gz`.

## Why this design is sound

Every production Ethereum zkVM (SP1, Risc0, OpenVM, ZisK, Pico, ZKsync
Airbender, EF's reference zkEVM, ...) proves block validation with
exactly this stack: `reth` + `revm` + an MPT-verified execution
witness. Verifying the pre-state against the parent block's
`stateRoot` and recomputing the post-state root closes the soundness
gap that a pure `revm` guest (with an untrusted account/storage dump)
would leave open. See
`~/Documents/Notes/jolt-zkvm-ethereum-stack-survey.md` for a
side-by-side comparison of every major zkVM's Ethereum stack.
