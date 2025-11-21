# degree2_sumcheck mul_unreduced speedups

- `cargo bench -p jolt-core --bench degree2_sumcheck --features challenge-254-bit` → unoptimized challenge encoding (plain mul + mul_unreduced columns marked “unopt”)
- `cargo bench --package jolt-core --bench degree2_sumcheck -- --feature challenge-254-bit` → optimized bench harness challenge encoding (columns marked “opt”)

All speedups are measured relative to the slowest configuration: plain multiplication with the unoptimized challenge encoding.

| vars | plain mul (unopt challenge) | plain mul (harness opt) | mul_unreduced (unopt challenge) | mul_unreduced (harness opt) |
| ---: | --------------------------: | -----------------------: | ------------------------------: | ---------------------------: |
| 14 | 1.1772 (1.00x / 0.00%) | 1.1121 (1.06x / 5.85%) | 1.0524 (1.12x / 11.86%) | 0.9759 (1.21x / 20.63%) |
| 16 | 2.9961 (1.00x / 0.00%) | 2.9382 (1.02x / 1.97%) | 2.5280 (1.19x / 18.52%) | 2.3093 (1.30x / 29.74%) |
| 18 | 9.9429 (1.00x / 0.00%) | 8.7562 (1.14x / 13.55%) | 7.8041 (1.27x / 27.41%) | 7.1531 (1.39x / 39.00%) |
| 20 | 34.6280 (1.00x / 0.00%) | 32.1950 (1.08x / 7.56%) | 30.7700 (1.13x / 12.54%) | 26.8110 (1.29x / 29.16%) |
| 22 | 134.9500 (1.00x / 0.00%) | 120.5100 (1.12x / 11.98%) | 112.7700 (1.20x / 19.67%) | 99.2010 (1.36x / 36.04%) |
| 24 | 539.2700 (1.00x / 0.00%) | 493.1200 (1.09x / 9.36%) | 444.0200 (1.21x / 21.45%) | 418.5700 (1.29x / 28.84%) |

## Challenge encodings and why there are two runs

- **Unoptimized 254-bit challenge (`--features challenge-254-bit`)**:
  - Here `Fr::Challenge` is `Mont254BitChallenge<Fr>` from `jolt-core/src/field/challenge/mont_ark_u254.rs`, which ranges over the full 254-bit BN254 scalar field without restriction.
  - This configuration is what the first `cargo bench -p jolt-core --bench degree2_sumcheck --features challenge-254-bit` run uses; in the table these measurements are the columns labeled “unopt challenge”.

- **Optimized 125-bit MontU128 challenge (default, no feature)**:
  - Here `Fr::Challenge` is `MontU128Challenge<Fr>` from `jolt-core/src/field/challenge/mont_ark_u128.rs`, which stores a 125-bit subset of the field (two least-significant bits zeroed) to enable faster multiplication.
  - This is what the second `cargo bench -p jolt-core --bench degree2_sumcheck` run measures; in the table these measurements are the columns labeled “harness opt”.

- **Four distinct configurations from two runs**:
  - Each run benchmarks both multiplication modes in `benches/degree2_sumcheck.rs`: baseline plain field multiplication and the `mul_unreduced` delayed-reduction path.
  - Combined with the two challenge encodings above, the pair of runs therefore measures 4 meaningful configurations:
    1. plain mul + unoptimized 254-bit challenge,
    2. `mul_unreduced` + unoptimized 254-bit challenge,
    3. plain mul + optimized MontU128 challenge,
    4. `mul_unreduced` + optimized MontU128 challenge.

## Latest raw median times (Criterion, ms)

Median times here are the middle value reported by Criterion in the `[low mid high]` interval for each benchmark, converted to milliseconds.

| vars | plain mul (unopt challenge) | mul_unreduced (unopt challenge) | plain mul (harness opt) | mul_unreduced (harness opt) |
| ---: | --------------------------: | ------------------------------: | -----------------------: | ---------------------------: |
| 14 | 0.9384 | 0.7975 | 0.9272 | 0.7614 |
| 16 | 2.5694 | 2.1491 | 2.4624 | 2.0593 |
| 18 | 8.0199 | 7.0277 | 7.7773 | 6.6361 |
| 20 | 29.6570 | 25.8220 | 29.2780 | 25.2580 |
| 22 | 113.5400 | 99.9930 | 110.7700 | 96.6690 |
| 24 | 451.0600 | 398.9100 | 459.5500 | 405.2600 |
| 26 | 2004.1000 | 1605.5000 | 2441.0000 | 1562.9000 |

## Single-threaded run (RAYON_NUM_THREADS=1, optimized challenge only)

Command: `RAYON_NUM_THREADS=1 cargo bench -p jolt-core --bench degree2_sumcheck` with the default MontU128 challenge encoding (no `challenge-254-bit` feature) and both multiplication modes enabled.

| vars | plain mul (ms) | mul_unreduced (ms) | speedup factor (mul_unreduced / plain) | speedup % |
| ---: | -------------: | -----------------: | --------------------------------------: | --------: |
| 14 | 1.0828 | 0.8516 | 1.2715x | 27.15% |
| 16 | 5.2203 | 3.9697 | 1.3150x | 31.50% |
| 18 | 20.8540 | 15.4390 | 1.3507x | 35.07% |
| 20 | 83.8720 | 62.7900 | 1.3358x | 33.58% |
| 22 | 338.0400 | 251.9600 | 1.3416x | 34.16% |
| 24 | 1391.8000 | 1031.3000 | 1.3496x | 34.96% |
| 26 | 5969.7000 | 4098.7000 | 1.4565x | 45.65% |

## Single-threaded run (RAYON_NUM_THREADS=1, unoptimized 254-bit challenge)

Command: `RAYON_NUM_THREADS=1 cargo bench -p jolt-core --bench degree2_sumcheck --features challenge-254-bit`, which switches `Fr::Challenge` to the full-width `Mont254BitChallenge<Fr>` while still running both multiplication modes.

| vars | plain mul (ms) | mul_unreduced (ms) | speedup factor (mul_unreduced / plain) | speedup % |
| ---: | -------------: | -----------------: | --------------------------------------: | --------: |
| 14 | 1.2039 | 0.9541 | 1.2618x | 26.18% |
| 16 | 5.5572 | 4.1842 | 1.3281x | 32.81% |
| 18 | 21.7400 | 16.1230 | 1.3484x | 34.84% |
| 20 | 86.5710 | 66.0340 | 1.3110x | 31.10% |
| 22 | 344.4000 | 259.4700 | 1.3273x | 32.73% |
| 24 | 1386.1000 | 1064.2000 | 1.3025x | 30.25% |
| 26 | 5951.3000 | 4464.4000 | 1.3331x | 33.31% |
