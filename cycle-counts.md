# Cycle Counts (recursion verify / fibonacci example)

All counts are taken from the `tracer::emulator::cpu` log lines emitted by
`start_cycle_tracking` / `end_cycle_tracking` in the RISC-V guest.

## Latest rerun (2026-01-17): `recursion-guest` verifies 1 embedded fibonacci proof (output = 1)

### Commands

- Baseline (no Grumpkin MSM feature):
  - `CARGO_NET_OFFLINE=true RUST_LOG=info cargo run --offline --release -p recursion -- trace --example fibonacci --workdir output --embed --disk`
- With provable Grumpkin MSM fast-path enabled in the guest:
  - `CARGO_NET_OFFLINE=true RUST_LOG=info JOLT_GUEST_EXTRA_FEATURES_PKG=recursion-guest JOLT_GUEST_EXTRA_FEATURES=grumpkin-msm-provable cargo run --offline --release -p recursion -- trace --example fibonacci --workdir output --embed --disk`

### Top-level markers (baseline)

| Op | RV64IMAC cycles | Virtual cycles |
| --- | ---:| ---:|
| deserialize preprocessing | 368,135,585 | 398,397,501 |
| deserialize count of proofs | 36 | 67 |
| deserialize proof | 213,282,528 | 232,486,952 |
| deserialize device | 846 | 2,239 |
| verification | 1,543,697,282 | 1,707,480,350 |

Trace length (reported): 2,338,367,342 cycles (virtual).

### Top-level markers (`grumpkin-msm-provable`)

| Op | RV64IMAC cycles | Virtual cycles |
| --- | ---:| ---:|
| deserialize preprocessing | 368,117,357 | 398,370,467 |
| deserialize count of proofs | 36 | 67 |
| deserialize proof | 213,268,631 | 232,488,863 |
| deserialize device | 841 | 2,234 |
| verification | 1,595,143,781 | 1,719,035,184 |

Trace length (reported): 2,349,897,044 cycles (virtual).

### Hyrax PCS opening verification (`HyraxOpeningProof::verify`)

Baseline:

| Op | RV64IMAC cycles | Virtual cycles |
| --- | ---:| ---:|
| hyrax_verify_eq_L | 826,101 | 826,115 |
| hyrax_verify_eq_R | 827,777 | 827,786 |
| hyrax_verify_normalize_row_commitments | 3,409,998 | 4,806,104 |
| hyrax_verify_msm_rows | 181,291,274 | 211,650,468 |
| hyrax_verify_msm_product | 273,047,859 | 299,052,340 |
| hyrax_verify_dot_product | 730,591 | 730,611 |
| hyrax_verify_total | 460,137,078 | 517,898,328 |

With `grumpkin-msm-provable`:

| Op | RV64IMAC cycles | Virtual cycles |
| --- | ---:| ---:|
| hyrax_verify_eq_L | 826,208 | 826,222 |
| hyrax_verify_eq_R | 827,951 | 827,960 |
| hyrax_verify_normalize_row_commitments | 3,315,857 | 4,711,964 |
| hyrax_verify_msm_rows | 195,704,391 | 203,553,222 |
| hyrax_verify_msm_product | 310,231,167 | 318,814,785 |
| hyrax_verify_dot_product | 730,590 | 730,610 |
| hyrax_verify_total | 511,639,567 | 529,469,592 |

Delta (virtual cycles):
- `hyrax_verify_msm_rows`: -8,097,246
- `hyrax_verify_msm_product`: +19,762,445
- `hyrax_verify_total`: +11,571,264

## Historical notes (may be stale)

Previous measurements and deeper breakdowns from earlier iterations live in git history; the numbers above
match the current Hyrax verifier implementation and the current `grumpkin-msm-provable` integration.
