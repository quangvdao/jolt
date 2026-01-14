# Spartan outer sumcheck benchmark results (sha2-chain)

Source: `/Users/quang.dao/.cursor/projects/Users-quang-dao-Documents-SNARKs-jolt-spartan/agent-tools/5e0f33cc-a22d-47d7-9fdf-6eb376794b6a.txt`

## Extracted timings
All times are Criterion-estimated \([low, mid, high]\). Units are preserved per value.

| N | outer-baseline | outer-current | outer-naive | outer-round-batched | outer-streaming |
|---|---|---|---|---|---|
| 8 | [25.391 ms, 26.023 ms, 26.782 ms] | [6.3943 ms, 6.6402 ms, 6.8664 ms] | [40.305 ms, 42.518 ms, 46.2 ms] | [53.193 ms, 54.094 ms, 55.026 ms] | [10.08 ms, 10.572 ms, 11.122 ms] |
| 16 | [42.82 ms, 51.162 ms, 66.494 ms] | [10.554 ms, 11.12 ms, 11.858 ms] | [69.761 ms, 72.028 ms, 73.895 ms] | [98.053 ms, 99.503 ms, 101.17 ms] | [17.318 ms, 19.3 ms, 22.825 ms] |
| 32 | [82.475 ms, 84.129 ms, 85.932 ms] | [17.682 ms, 17.976 ms, 18.292 ms] | [137.4 ms, 144.48 ms, 155.65 ms] | [193.08 ms, 197.66 ms, 204.38 ms] | [30.409 ms, 31.308 ms, 32.468 ms] |
| 64 | [165.48 ms, 170.94 ms, 178.31 ms] | [30.815 ms, 32.163 ms, 34.166 ms] | [252.73 ms, 264.4 ms, 281.37 ms] | [378.18 ms, 383.84 ms, 390.4 ms] | [54.881 ms, 60.021 ms, 67.324 ms] |
| 128 | [321.92 ms, 335.97 ms, 351.53 ms] | [54.747 ms, 55.823 ms, 56.964 ms] | [480.63 ms, 505.73 ms, 538.46 ms] | [753.88 ms, 764.4 ms, 775.99 ms] | [103.23 ms, 105.07 ms, 107.33 ms] |
| 256 | [627.11 ms, 654.4 ms, 680.66 ms] | [104.13 ms, 119.41 ms, 137.17 ms] | [881.52 ms, 913.68 ms, 969.56 ms] | [1.4586 s, 1.4735 s, 1.4888 s] | [188.71 ms, 194.17 ms, 199.71 ms] |
| 512 | [1.6082 s, 1.6828 s, 1.7487 s] | [186.7 ms, 191.67 ms, 198.04 ms] | [1.9355 s, 2.0334 s, 2.1354 s] | [3.0438 s, 3.2118 s, 3.4154 s] | [413.64 ms, 430.21 ms, 449.81 ms] |
| 1024 | [4.1106 s, 5.4541 s, 7.118 s] | [381.75 ms, 395.33 ms, 407.65 ms] | [3.5241 s, 3.8303 s, 4.2744 s] | [6.06 s, 7.1714 s, 8.941 s] | [763.89 ms, 780.68 ms, 797.11 ms] |
| 2048 | [4.9767 s, 5.0486 s, 5.1348 s] | [897.11 ms, 1.0206 s, 1.156 s] | [6.8162 s, 6.993 s, 7.1734 s] | [11.983 s, 12.045 s, 12.111 s] | [1.4948 s, 1.5315 s, 1.577 s] |

## Relative to `outer-current` (midpoint ratios)
Ratios computed as `(variant_mid / outer-current_mid)` after converting each midpoint to milliseconds.

| N | outer-baseline | outer-naive | outer-round-batched | outer-streaming |
|---|---|---|---|---|
| 8 | 3.92× | 6.40× | 8.15× | 1.59× |
| 16 | 4.60× | 6.48× | 8.95× | 1.74× |
| 32 | 4.68× | 8.04× | 11.00× | 1.74× |
| 64 | 5.31× | 8.22× | 11.93× | 1.87× |
| 128 | 6.02× | 9.06× | 13.69× | 1.88× |
| 256 | 5.48× | 7.65× | 12.34× | 1.63× |
| 512 | 8.78× | 10.61× | 16.76× | 2.24× |
| 1024 | 13.80× | 9.69× | 18.14× | 1.97× |
| 2048 | 4.95× | 6.85× | 11.80× | 1.50× |

## Scaling (midpoint per doubling)
For each variant, this is `mid(N*2) / mid(N)` (midpoints converted to ms).

- **outer-baseline**: 16:1.97×, 32:1.64×, 64:2.03×, 128:1.97×, 256:1.95×, 512:2.57×, 1024:3.24×, 2048:0.93× (geo-mean 1.93×)
- **outer-current**: 16:1.67×, 32:1.62×, 64:1.79×, 128:1.74×, 256:2.14×, 512:1.61×, 1024:2.06×, 2048:2.58× (geo-mean 1.88×)
- **outer-naive**: 16:1.69×, 32:2.01×, 64:1.83×, 128:1.91×, 256:1.81×, 512:2.23×, 1024:1.88×, 2048:1.83× (geo-mean 1.89×)
- **outer-round-batched**: 16:1.84×, 32:1.99×, 64:1.94×, 128:1.99×, 256:1.93×, 512:2.18×, 1024:2.23×, 2048:1.68× (geo-mean 1.97×)
- **outer-streaming**: 16:1.83×, 32:1.62×, 64:1.92×, 128:1.75×, 256:1.85×, 512:2.22×, 1024:1.81×, 2048:1.96× (geo-mean 1.86×)

## Analysis
- **Overall ordering (midpoint)**: `outer-current` is consistently fastest; `outer-streaming` is ~1.6–2.2× slower; `outer-baseline` ~4–14× slower; `outer-naive` ~6–11× slower; `outer-round-batched` ~8–18× slower in this run.
- **Scaling**: For `outer-current` and `outer-streaming`, per-doubling multipliers are typically ~1.6–2.2× (sub-quadratic in N over this range). The baseline/naive/round-batched variants grow faster and become multi-second beyond N≥512.
- **Noise / methodology**: Many cases show Criterion warnings about not completing 10 samples in the requested 1s; Criterion increased the effective measurement time. Use these numbers as a quick smoke-test, not a stable perf regression signal.

## Notes / caveats
- Criterion output sometimes mixes units within a single bracket (e.g. `[897 ms, 1.02 s, 1.15 s]`); this report preserves units per value and converts per-value for ratios.

# Spartan outer sumcheck benchmark results (sha2-chain)

Source: `/Users/quang.dao/.cursor/projects/Users-quang-dao-Documents-SNARKs-jolt-spartan/agent-tools/5e0f33cc-a22d-47d7-9fdf-6eb376794b6a.txt`

## Extracted timings
All times are Criterion-estimated \([low, mid, high]\). Units are preserved per value.

| N | outer-baseline | outer-current | outer-naive | outer-round-batched | outer-streaming |
|---|---|---|---|---|---|
| 8 | [25.391 ms, 26.023 ms, 26.782 ms] | [6.3943 ms, 6.6402 ms, 6.8664 ms] | [40.305 ms, 42.518 ms, 46.2 ms] | [53.193 ms, 54.094 ms, 55.026 ms] | [10.08 ms, 10.572 ms, 11.122 ms] |
| 16 | [42.82 ms, 51.162 ms, 66.494 ms] | [10.554 ms, 11.12 ms, 11.858 ms] | [69.761 ms, 72.028 ms, 73.895 ms] | [98.053 ms, 99.503 ms, 101.17 ms] | [17.318 ms, 19.3 ms, 22.825 ms] |
| 32 | [82.475 ms, 84.129 ms, 85.932 ms] | [17.682 ms, 17.976 ms, 18.292 ms] | [137.4 ms, 144.48 ms, 155.65 ms] | [193.08 ms, 197.66 ms, 204.38 ms] | [30.409 ms, 31.308 ms, 32.468 ms] |
| 64 | [165.48 ms, 170.94 ms, 178.31 ms] | [30.815 ms, 32.163 ms, 34.166 ms] | [252.73 ms, 264.4 ms, 281.37 ms] | [378.18 ms, 383.84 ms, 390.4 ms] | [54.881 ms, 60.021 ms, 67.324 ms] |
| 128 | [321.92 ms, 335.97 ms, 351.53 ms] | [54.747 ms, 55.823 ms, 56.964 ms] | [480.63 ms, 505.73 ms, 538.46 ms] | [753.88 ms, 764.4 ms, 775.99 ms] | [103.23 ms, 105.07 ms, 107.33 ms] |
| 256 | [627.11 ms, 654.4 ms, 680.66 ms] | [104.13 ms, 119.41 ms, 137.17 ms] | [881.52 ms, 913.68 ms, 969.56 ms] | [1.4586 s, 1.4735 s, 1.4888 s] | [188.71 ms, 194.17 ms, 199.71 ms] |
| 512 | [1.6082 s, 1.6828 s, 1.7487 s] | [186.7 ms, 191.67 ms, 198.04 ms] | [1.9355 s, 2.0334 s, 2.1354 s] | [3.0438 s, 3.2118 s, 3.4154 s] | [413.64 ms, 430.21 ms, 449.81 ms] |
| 1024 | [4.1106 s, 5.4541 s, 7.118 s] | [381.75 ms, 395.33 ms, 407.65 ms] | [3.5241 s, 3.8303 s, 4.2744 s] | [6.06 s, 7.1714 s, 8.941 s] | [763.89 ms, 780.68 ms, 797.11 ms] |
| 2048 | [4.9767 s, 5.0486 s, 5.1348 s] | [897.11 ms, 1.0206 s, 1.156 s] | [6.8162 s, 6.993 s, 7.1734 s] | [11.983 s, 12.045 s, 12.111 s] | [1.4948 s, 1.5315 s, 1.577 s] |

## Relative to `outer-current` (midpoint ratios)
Ratios computed as `(variant_mid / outer-current_mid)` after converting each midpoint to milliseconds.

| N | outer-baseline | outer-naive | outer-round-batched | outer-streaming |
|---|---|---|---|---|
| 8 | 3.92× | 6.40× | 8.15× | 1.59× |
| 16 | 4.60× | 6.48× | 8.95× | 1.74× |
| 32 | 4.68× | 8.04× | 11.00× | 1.74× |
| 64 | 5.31× | 8.22× | 11.93× | 1.87× |
| 128 | 6.02× | 9.06× | 13.69× | 1.88× |
| 256 | 5.48× | 7.65× | 12.34× | 1.63× |
| 512 | 8.78× | 10.61× | 16.76× | 2.24× |
| 1024 | 13.80× | 9.69× | 18.14× | 1.97× |
| 2048 | 4.95× | 6.85× | 11.80× | 1.50× |

## Scaling (midpoint per doubling)
For each variant, this is `mid(N*2) / mid(N)` (midpoints converted to ms).

- **outer-baseline**: 16:1.97×, 32:1.64×, 64:2.03×, 128:1.97×, 256:1.95×, 512:2.57×, 1024:3.24×, 2048:0.93× (geo-mean 1.93×)
- **outer-current**: 16:1.67×, 32:1.62×, 64:1.79×, 128:1.74×, 256:2.14×, 512:1.61×, 1024:2.06×, 2048:2.58× (geo-mean 1.88×)
- **outer-naive**: 16:1.69×, 32:2.01×, 64:1.83×, 128:1.91×, 256:1.81×, 512:2.23×, 1024:1.88×, 2048:1.83× (geo-mean 1.89×)
- **outer-round-batched**: 16:1.84×, 32:1.99×, 64:1.94×, 128:1.99×, 256:1.93×, 512:2.18×, 1024:2.23×, 2048:1.68× (geo-mean 1.97×)
- **outer-streaming**: 16:1.83×, 32:1.62×, 64:1.92×, 128:1.75×, 256:1.85×, 512:2.22×, 1024:1.81×, 2048:1.96× (geo-mean 1.86×)

## Analysis
- **Overall ordering (midpoint)**: `outer-current` is consistently fastest; `outer-streaming` is ~1.6–2.2× slower; `outer-baseline` ~4–14× slower; `outer-naive` ~6–11× slower; `outer-round-batched` ~8–18× slower in this run.
- **Scaling**: For `outer-current` and `outer-streaming`, per-doubling multipliers are typically ~1.6–2.2× (sub-quadratic in N over this range). The baseline/naive/round-batched variants grow faster and become multi-second beyond N≥512.
- **Noise / methodology**: Many cases show Criterion warnings about not completing 10 samples in the requested 1s; Criterion increased the effective measurement time. Use these numbers as a quick smoke-test, not a stable perf regression signal.

## Notes / caveats
- Criterion output sometimes mixes units within a single bracket (e.g. `[897 ms, 1.02 s, 1.15 s]`); this report preserves units per value and converts per-value for ratios.

