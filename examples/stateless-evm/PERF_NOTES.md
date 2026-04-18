# Stateless EVM: Trace-Cost Experiments

Notes from instrumenting the `stateless-evm` example to understand where Jolt's
expanded cycles go on a real Ethereum mainnet block.

All numbers are from the canonical end-to-end fixture
`examples/stateless-evm/fixtures/mainnet_block_22974576.json` (block 22,974,576,
115 transactions, 7.6 M gas), unless stated otherwise.

Guest is built with `cargo install --path .` (jolt CLI) targeting
`riscv64imac-zero-linux-musl`, profile `guest` (inherits `release`,
`lto = "fat"`, `codegen-units = 1`).
Set `JOLT_BACKTRACE=1` when you want the ELF's `.symtab` preserved for
function-level attribution (disabled by default to shrink the ELF).

## 1. Vocabulary

Before presenting numbers, a glossary of the terms the host logs:

- **Static raw bytecode instructions**: count of instructions in the ELF as
  produced by `tracer::decode` (see `tracer/src/lib.rs:655`), before any
  virtual-sequence expansion.
  For this guest: 775,181.
- **Static bytecode instructions**: after Jolt's `guest::program::decode` has
  pre-expanded every virtual sequence into its constituent helpers.
  For this guest: 1,334,802.
- **Raw RISC-V cycles**: number of trace cycles where `Instruction::is_real()`
  returns true.
  Counts ordinary RISC-V ops plus one "anchor" per executed virtual sequence.
  Definition at `tracer/src/instruction/mod.rs:842`.
- **Expanded cycles**: total length of the execution trace.
  Every helper inside a virtual sequence shows up as one cycle here.
  This is the metric Jolt's prover actually pays for.
- **Expansion factor**: `expanded_cycles / raw_riscv_cycles`.
- **Anchor**: the instruction whose opcode matches the original high-level
  bytecode entry.
  In the trace it carries `virtual_sequence_remaining = Some(0)`.
- **Helper**: a sub-instruction produced by expanding a virtual sequence.
  Carries `virtual_sequence_remaining = Some(k)` with `k > 0`.
- **Virtual\* instructions** (e.g. `VirtualMULI`, `VirtualPow2`, `VirtualROTRI`):
  Jolt-internal opcodes emitted exclusively as helpers of real RISC-V
  instructions that don't have a 1:1 constraint encoding.
- **`INLINE:<NAME>`**: a Jolt inline, i.e. a registered custom opcode
  (`tracer::list_registered_inlines`) that expands into its own helper
  sequence.
  Current block sees `KECCAK256_INLINE`, `SECP256K1_DIVQ`, `SECP256K1_SQUAREQ`,
  `SECP256K1_MULQ`.

## 2. Trace summary (mainnet block 22,974,576, inline Keccak backend)

```
static_raw_bytecode_instructions = 775,181
static_bytecode_instructions     = 1,334,802
static_compressed_bytecode       = 350,198

raw_riscv_cycles    = 152,840,093
expanded_cycles     = 510,295,058
inline_helper_cycles= 357,454,965
expansion_factor    = 3.3388
```

Category split:

| Category | Raw cycles | Raw % | Expanded cycles | Expanded % |
|---|---:|---:|---:|---:|
| `inline_helper` | 0 | 0.00 % | 357,454,965 | 70.05 % |
| `alu_misc` | 53,504,679 | 35.01 % | 53,504,679 | 10.49 % |
| `inline_anchor` | 40,607,428 | 26.57 % | 40,607,428 | 7.96 % |
| `load_store` | 35,607,546 | 23.30 % | 35,607,546 | 6.98 % |
| `control_flow` | 23,120,440 | 15.13 % | 23,120,440 | 4.53 % |

Top instructions by expanded-cycle share (top 12):

| Instruction | Expanded cycles | % |
|---|---:|---:|
| `XOR` | 78,690,639 | 15.42 % |
| `ADDI` | 54,424,677 | 10.67 % |
| `LD` | 45,086,842 | 8.84 % |
| `MUL` | 34,658,905 | 6.79 % |
| `VirtualMULI` | 34,658,113 | 6.79 % |
| `VirtualPow2` | 32,037,404 | 6.28 % |
| `ANDI` | 29,930,844 | 5.87 % |
| `SD` | 26,873,733 | 5.27 % |
| `VirtualROTRI` | 23,448,240 | 4.60 % |
| `ANDN` | 20,214,000 | 3.96 % |
| `VirtualSRLI` | 19,921,936 | 3.90 % |
| `XORI` | 16,774,855 | 3.29 % |

## 3. Helpers by source (top 12)

The `trace helpers by source` logger groups every helper cycle by the
raw-bytecode opcode that generated it.
Attribution uses `tracer::decode`'s unexpanded opcode stream and matches by
PC (`build_bytecode_source_labels` in `examples/stateless-evm/host/src/main.rs`).

| Source | Helpers | % of 357.45 M helpers | Anchors | Avg helpers/anchor |
|---|---:|---:|---:|---:|
| `INLINE:KECCAK256_INLINE` | 109,795,710 | 30.72 % | 33,690 | 3,259 |
| `LBU` | 97,879,103 | 27.38 % | 13,982,729 | 7 |
| `SB` | 77,679,948 | 21.73 % | 6,473,329 | 12 |
| `SW` | 26,497,590 | 7.41 % | 1,892,685 | 14 |
| `LW` | 12,303,396 | 3.44 % | 1,757,628 | 7 |
| `INLINE:SECP256K1_DIVQ` | 6,453,148 | 1.81 % | 33,436 | 193 |
| `INLINE:SECP256K1_SQUAREQ` | 6,135,194 | 1.72 % | 36,959 | 166 |
| `LHU` | 3,853,440 | 1.08 % | 481,680 | 8 |
| `INLINE:SECP256K1_MULQ` | 3,640,245 | 1.02 % | 19,677 | 185 |
| `SH` | 2,749,175 | 0.77 % | 211,475 | 13 |
| `SRLIW` | 2,471,380 | 0.69 % | 1,235,690 | 2 |
| `LB` | 1,446,095 | 0.40 % | 206,585 | 7 |

Observation: after Keccak, the dominant helper source is not cryptography but
the seven sub-word load/store opcodes (`LBU`, `SB`, `SW`, `LW`, `LHU`, `SH`,
`LB`). Together they account for 223 M helper cycles (62.4 % of all helpers).

Expansion sequences for these live in
`tracer/src/instruction/{lbu,lb,lhu,lh,lw,sb,sh,sw}.rs`.
Example: `LBU` on RV64 expands to `ADDI, ANDI, LD, XORI, SLLI, SLL, SRLI`
because Jolt's RAM is 8-byte aligned only
(`tracer/src/instruction/lbu.rs:110-125`).

## 4. Keccak A/B (inline vs. software) end-to-end

The guest exposes a `software-keccak` feature flag
(`examples/stateless-evm/guest/Cargo.toml`) that swaps
`jolt_inlines_keccak256::Keccak256::digest` for `alloy_primitives::Keccak256`.
Host CLI selects via `--keccak-backend {inline,software}`.

### Small fixture (`ether_transfers_osaka_1M.json`, 47 transfers)

| Backend | Expanded cycles | Raw cycles | Inline helper cycles |
|---|---:|---:|---:|
| Inline | 17,568,345 | 4,955,091 | 12,613,254 |
| Software | 19,533,700 | 6,853,030 | 12,680,670 |
| **Delta** | **-1,965,355 (-10.06 %)** | -1,897,939 | -67,416 |

### Mainnet block 22,974,576 (115 txs, 7.6 M gas)

| Backend | Expanded cycles | Raw cycles | Inline helper cycles |
|---|---:|---:|---:|
| Inline | 510,295,058 | 152,840,093 | 357,454,965 |
| Software | 668,704,828 | 188,567,415 | 480,137,413 |
| **Delta** | **-158,409,770 (-23.69 %)** | -35,727,322 | -122,682,448 |

## 5. Per-call Keccak cost (hash-bench)

Added `JOLT_ANALYZE_ONLY=1` to `examples/hash-bench/src/main.rs` so the hash
benchmark can be traced without the expensive prove step.
The guest already wraps every size/variant call in
`start_cycle_tracking` / `end_cycle_tracking`
(see `examples/hash-bench/guest/src/lib.rs:61-68`), which emits per-call
markers that `tracer::emulator::cpu::handle_jolt_cycle_marker` logs.

Raw data (sha3 reference vs. `jolt_inlines_keccak256`):

| Input (B) | 136-B blocks | SW total | Inline total | Cycles saved | Speedup | SW cyc/B | Inline cyc/B |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 7,536 | 3,635 | 3,901 | 2.07x | 235.5 | 113.6 |
| 64 | 1 | 7,544 | 3,655 | 3,889 | 2.06x | 117.9 | 57.1 |
| 128 | 1 | 7,669 | 3,695 | 3,974 | 2.08x | 59.9 | 28.9 |
| 136 | 2 | 15,831 | 6,916 | 8,915 | 2.29x | 116.4 | 50.9 |
| 256 | 2 | 15,876 | 7,027 | 8,849 | 2.26x | 62.0 | 27.4 |
| 512 | 4 | 32,480 | 13,691 | 18,789 | 2.37x | 63.4 | 26.7 |
| 1024 | 8 | 65,688 | 27,019 | 38,669 | 2.43x | 64.1 | 26.4 |
| 2048 | 16 | 132,284 | 53,855 | 78,429 | 2.46x | 64.6 | 26.3 |

Fitted per-block cost (slope of total-cycles vs. n\_blocks):

- Software (`sha3::Keccak256`): ~8,300 cycles per 136-byte absorption block
  (~61 cyc/byte in the limit).
- Inline (`jolt_inlines_keccak256::Keccak256`): ~3,360 cycles per block
  (one `f1600` permutation = one `KECCAK256_INLINE` anchor + ~3,259 helpers,
  plus ~300 cycles fixed call overhead).
- **Per-block speedup: 2.47x.**

Almost all inline cost lives in the virtual-instruction stream.
The inline's "raw RV64 cycles" column stays near 400 cycles regardless of
input size; the hashing work happens entirely inside the expanded helper
sequence of each `KECCAK256_INLINE` anchor.

## 6. Keccak input-size histogram (mainnet)

From the per-call histogram in `CryptoTraceStats::keccak_size_hist_136`
(bucketed by 136-byte Keccak rate, see
`examples/stateless-evm/guest/src/lib.rs:232-246`):

| Bucket | Calls | % |
|---|---:|---:|
| 0..136 | 5,874 | 54.0 % |
| 136..272 | 458 | 4.2 % |
| 272..408 | 736 | 6.8 % |
| 408..544 | 3,639 | 33.4 % |
| 544..680 | 10 | 0.1 % |
| 680..816 | 7 | 0.1 % |
| 816..952 | 6 | 0.1 % |
| >=952 | 153 | 1.4 % |

Total: 10,883 calls, 3,928,215 input bytes, avg 360.95 bytes/call.

Reconciling with the per-block cost: expected inline Keccak cycles ≈
`5,874 × 3,700 + 458 × 7,000 + 736 × 10,300 + 3,639 × 13,700 + tail ≈ 107 M`.
Measured = 109.8 M helpers + 33.7 k anchors ≈ 110 M. Matches within 3 %.

Expected software Keccak cycles via 8,300 cyc/block ≈ 250-270 M, so the A/B
saving of ~158 M expanded cycles is consistent with per-call microbench
data and the histogram.

## 7. Helpers by containing function (mainnet)

Added `log_memops_by_function` in
`examples/stateless-evm/host/src/main.rs` which resolves the PC of each
sub-word memop anchor to its containing ELF symbol via
`object::File::parse` + `rustc_demangle`.

Requires `JOLT_BACKTRACE=1` to preserve `.symtab` in the guest ELF (see
`src/main.rs:194-242` where the jolt CLI conditionally passes
`-Cstrip=symbols`).

Total sub-word memop cycles on mainnet:
**25.1 M anchors + 223.1 M helpers = 248.2 M expanded cycles (48.6 % of trace)**.

Top 20 functions:

| Rank | Cycles | % of subword | % of trace | Function |
|---:|---:|---:|---:|---|
| 1 | 95,778,451 | 38.59 % | 18.77 % | `memcpy` |
| 2 | 15,618,624 | 6.29 % | 3.06 % | `revm_interpreter::instructions::stack::swap::<1>` |
| 3 | 15,328,604 | 6.18 % | 3.00 % | `zeth_mpt::mpt::node::Node::decode` (alloy-rlp Decodable) |
| 4 | 15,068,144 | 6.07 % | 2.95 % | `zeth_mpt::mpt::node::Node::resolve_digests` |
| 5 | 14,272,944 | 5.75 % | 2.80 % | `memcmp` |
| 6 | 9,818,974 | 3.96 % | 1.92 % | `memset` |
| 7 | 8,242,752 | 3.32 % | 1.62 % | `revm_interpreter::instructions::stack::swap::<2>` |
| 8 | 8,026,534 | 3.23 % | 1.57 % | `enframe` (zeroos allocator) |
| 9 | 7,451,564 | 3.00 % | 1.46 % | `__libc_free` |
| 10 | 6,386,503 | 2.57 % | 1.25 % | `revm_bytecode::legacy::raw::LegacyRawBytecode::into_analyzed` |
| 11 | 4,096,512 | 1.65 % | 0.80 % | `swap::<3>` |
| 12 | 4,056,284 | 1.63 % | 0.80 % | `alloy_evm::eth::EthEvm::transact` |
| 13 | 3,818,752 | 1.54 % | 0.75 % | `foldhash::hash_bytes_long` |
| 14 | 3,707,132 | 1.49 % | 0.73 % | `get_meta` (allocator metadata) |
| 15 | 2,557,786 | 1.03 % | 0.50 % | `__libc_malloc_impl` |
| 16 | 2,457,728 | 0.99 % | 0.48 % | `revm_interpreter::instructions::memory::mstore` |
| 17 | 1,853,376 | 0.75 % | 0.36 % | `swap::<4>` |
| 18 | 1,803,741 | 0.73 % | 0.35 % | `alloc_slot` |
| 19 | 1,466,582 | 0.59 % | 0.29 % | `size_to_class` |
| 20 | 1,272,482 | 0.51 % | 0.25 % | postcard `Vec<Bytes>::deserialize` |

### Cluster summary

| Cluster | Ranks | Sub-word cycles | % of full trace |
|---|---|---:|---:|
| C library (`memcpy`, `memcmp`, `memset`) | 1, 5, 6 | 119.87 M | **23.5 %** |
| EVM interpreter (`swap1-4`, `mstore`, `transact`) | 2, 7, 11, 12, 16, 17 | 36.43 M | 7.1 % |
| MPT + RLP + bytecode analysis | 3, 4, 10, 13, 20 | 44.08 M | 8.6 % |
| Allocator (`enframe`, `__libc_free`, `get_meta`, `__libc_malloc_impl`, `alloc_slot`, `size_to_class`) | 8, 9, 14, 15, 18, 19 | 25.01 M | 4.9 % |

The `memcpy` attribution is a byte-by-byte RMW loop: 3.30 M `SB` anchors paired
with 3.30 M `LBU` anchors, plus 1.15 M `SW` / 1.17 M `LW` anchors for aligned
fast-paths.
That pattern is `compiler_builtins`'s generic `mem::copy_forward` fallback,
which is what Rust emits for RV64 `riscv64imac-zero-linux-musl` in the absence
of a libc memcpy.

## 8. Inline coverage wired up

For reference, the `stateless-evm` guest currently routes the following
cryptographic operations through Jolt inlines:

| Operation | Library call | Inline crate |
|---|---|---|
| Keccak256 | `alloy_primitives::keccak256` | `jolt-inlines-keccak256` |
| SHA-256 | `revm-precompile::sha256::ripemd160_run` path | `jolt-inlines-sha2` |
| secp256k1 `ecrecover` | `revm-precompile::secp256k1::ecrecover` | `jolt-inlines-secp256k1` |
| secp256k1 tx-signer recovery | `alloy-consensus::SignerRecoverable` | `jolt-inlines-secp256k1` |
| p256r1 signature verify | `revm-precompile::secp256r1::p256_verify` | `jolt-inlines-p256` |

Wiring lives in `examples/stateless-evm/guest/src/lib.rs`
(`JoltK256Provider`, `JoltRevmCrypto`, `install_jolt_crypto`).

## 9. Directions considered

Preference established during this session: **no new inlines, `unsafe` is
acceptable inside shim crates when unavoidable with clear SAFETY comments,
public helpers expose safe APIs**.
Under those constraints the clear leverage points, in order of expected
impact and feasibility, are:

1. **Memops crate + local `revm-interpreter` hot-path patch (shipped).**
   See section 10.
   Current default saves ~45.8 M expanded cycles (~9.0 %) on the real
   mainnet block relative to the original baseline
   (`510.28 M -> 464.50 M`).
   Of that, the `revm` hot-path patch contributes ~34.4 M cycles on top of
   the memops-only build (`498.93 M -> 464.50 M`).
   The cost is a ~0.48 M regression (~2.7 %) on the small
   `ether_transfers_osaka_1M.json` fixture, so this is a deliberate
   optimization for the mainnet-sized target workload.

2. **RLP / MPT decoder reshape.**
   Rewrite `zeth_mpt::mpt::node::Node::decode` and `resolve_digests` to read
   whole `u64` chunks where the node geometry permits (branch-node child
   hashes are 4 x u64, leaf value lengths are RLP-prefixed but 32-byte hashes
   are a common fast path).
   Medium effort; addresses ranks 3, 4, 10 (~36.8 M cycles).

3. **Byte-array -> word-array representation in remaining revm structures.**
   Represent `U256` stack slots and `SharedMemory` as `[u64; N]` with byte
   views for EVM semantics.
   Big upstream surface, meaningful payoff. After the local
   `revm-interpreter` patch, the next largest memop rows are no longer stack
   swaps or `mstore`; the remaining pressure is mostly MPT decoding,
   allocation, and generic bytecode analysis.

4. **Shifted-write `memcpy` override.**
   Match or beat `compiler_builtins`' mismatched-alignment path (LW on the
   4-byte-aligned source into aligned SW writes through a moving shift/OR
   window). Harder to get right than `memset`; only worth doing once we
   have tests that cover all alignment corners.

5. **Allocator.** Partly out of scope unless swapping `zeroos_runtime_musl`
   for a simpler bump allocator in this example.

## 10. `memops` crate and `revm` hot-path patch

`examples/stateless-evm/memops/` is a small `no_std` crate that installs
`#[no_mangle] memset` and `memcmp` (RV64IMAC only) and exposes safe helpers
(`copy_words` / `zero_words` / `swap_words` / `cmp_words`) in `crate::safe`.

### Measurement matrix

We measured seven variants on the mainnet fixture:

| Variant | `memcpy` cycles | `memcmp` cycles | `memset` cycles | Total expanded |
|---|---:|---:|---:|---:|
| Baseline (`compiler_builtins`) | 95.78 M | 14.27 M | 9.82 M | 510.28 M |
| Naive overrides for all four   | 144.97 M | 2.91 M | - | 560.14 M |
| Staged 8/4/2-byte overrides    | 120.83 M | 2.91 M | - | 534.90 M |
| Alignment-minimised overrides  | 119.62 M | 2.91 M | - | 533.25 M |
| `memset` override only         | 95.78 M | 14.27 M | ~1.4 M | 507.16 M |
| `memset` + full-reassembly `memcmp` | 95.78 M | 1.29 M | ~1.4 M | 497.36 M |
| `memset` + shared-alignment `memcmp` | 95.78 M | 2.88 M | ~1.4 M | 498.93 M |
| **Shipped: memops + `revm` hot-path patch** | **95.78 M** | **2.88 M** | **~1.4 M** | **464.50 M** |

Interpretation:

- `compiler_builtins`'s `memcpy` runs a classic shifted-write algorithm
  (`LW` on a 4-byte-aligned source through a moving shift/OR window into
  aligned `SW` writes) that keeps mismatched-alignment copies in the
  word-sized path.
  A naive replacement that falls back to byte copies whenever
  `(src ^ dst) & 7 != 0` regresses memcpy by ~25 M expanded cycles
  (+26 %) on this block, because the `(src ^ dst) & 7` test fires often
  in MPT / RLP / EVM memory paths.
- `compiler_builtins`'s `memcmp` is a byte loop.
  A full word-reassembly `memcmp` buys the biggest mainnet win
  (`510.28 M -> 497.36 M` total expanded), but it regressed the small
  EF transfer fixture from `17.70 M` to `18.22 M` cycles because the
  setup cost is too eager on shorter compares.
  The current shipped compromise only takes the word path when the two
  pointers share the same 8-byte alignment.
  That still cuts the mainnet `memcmp` hotspot from `14.27 M` down to
  `2.88 M` while landing at `18.18 M` on the small fixture.
- `memset` has no second pointer so the alignment-matching concern does
  not apply. Our `SD`-per-8-byte loop straightforwardly beats
  `compiler_builtins`' mixed `SW`/`SB` loop.
- The local `revm-interpreter` patch then attacks the fixed-width `U256`
  movers directly:
  - `Stack::dup` now copies four `u64` limbs explicitly instead of lowering a
    single `U256` copy through `ptr::copy_nonoverlapping`.
  - `Stack::exchange` now swaps four `u64` limbs explicitly instead of
    lowering through `ptr::swap_nonoverlapping`.
  - `MemoryTr` gained `set_u256_be`, and `SharedMemory` overrides it with an
    aligned four-word store path. `mstore` now calls that directly.
  On the mainnet fixture this drops total subword-memop cycles from
  `227.29 M` to `192.57 M`, and the old `swap::<1>`, `swap::<2>`,
  `swap::<3>`, and `mstore` rows disappear from the top-20 memop table.

Net shipped win on mainnet: **-45.79 M expanded cycles (-9.0 %)** relative
to the original baseline, clean clippy, with a **+0.48 M expanded-cycle
(+2.7 %)** regression on the small EF fixture.

## 11. Instrumentation deltas (reference)

Files touched to produce these numbers:

- `examples/stateless-evm/guest/src/lib.rs`: `CryptoTraceStats`,
  `keccak_size_hist_136` histogram, `software-keccak` feature gate.
- `examples/stateless-evm/guest/Cargo.toml`: `software-keccak` feature.
- `examples/stateless-evm/host/Cargo.toml`: added `tracer`, `object`,
  `rustc-demangle`.
- `examples/stateless-evm/host/src/main.rs`:
  - `load_guest_binary` / `load_function_symbols` / `lookup_function_id`
    (ELF symbol table parsing).
  - `log_trace_stats`, `log_opcode_split`, `log_helpers_by_source`,
    `log_memops_by_function`, `log_crypto_stats`.
  - `--keccak-backend {inline,software}` CLI flag.
- `examples/hash-bench/src/main.rs`: `JOLT_ANALYZE_ONLY` fast-path that skips
  prove/verify for cheap per-call cycle measurement.
- `patches/revm-interpreter/`: local crates.io override carrying the
  `Stack::{dup,exchange}` and `MemoryTr::set_u256_be` hot-path patch.
- `Cargo.toml` (workspace):
  - `rustc-demangle = "0.1"` added.
  - `[patch.crates-io] revm-interpreter = { path = "patches/revm-interpreter" }`
    added.

Reproduce from branch tip:

```bash
# End-to-end mainnet trace with all attributions (requires ~1 min compile
# the first time, ~4 min trace).
JOLT_BACKTRACE=1 RUST_LOG=info cargo run --release -p stateless-evm --quiet -- \
  --mode analyze --keccak-backend inline \
  examples/stateless-evm/fixtures/mainnet_block_22974576.json

# Per-call Keccak cycle microbench (~2 min):
JOLT_ANALYZE_ONLY=1 RUST_LOG=info cargo run --release -p hash-bench --quiet

# Keccak A/B on mainnet (run twice, inline then software):
cargo run --release -p stateless-evm --quiet -- --mode analyze \
  --keccak-backend software \
  examples/stateless-evm/fixtures/mainnet_block_22974576.json
```
