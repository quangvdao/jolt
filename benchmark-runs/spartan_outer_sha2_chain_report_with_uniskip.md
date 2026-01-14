# Spartan outer sumcheck benchmark results (sha2-chain) — run with `outer-uni-skip`

Source: `/Users/quang.dao/.cursor/projects/Users-quang-dao-Documents-SNARKs-jolt-spartan/agent-tools/1d46f132-7838-4bf3-8b82-d6d33504b7ae.txt`

## Extracted timings
All times are Criterion-estimated \([low, mid, high]\). Units are preserved per value.

| N | outer-current | outer-uni-skip | outer-streaming | outer-round-batched | outer-baseline | outer-naive |
|---|---|---|---|---|---|---|
| 8 | [6.3808 ms, 6.5957 ms, 6.8447 ms] | [6.0203 ms, 6.2032 ms, 6.4074 ms] | [10.271 ms, 10.924 ms, 11.549 ms] | [53.438 ms, 54.685 ms, 55.987 ms] | [24.565 ms, 25.102 ms, 25.672 ms] | [35.054 ms, 38.513 ms, 43.12 ms] |
| 16 | [10.303 ms, 10.629 ms, 10.974 ms] | [8.9795 ms, 9.2142 ms, 9.4728 ms] | [16.121 ms, 16.803 ms, 17.472 ms] | [92.531 ms, 96.109 ms, 99.836 ms] | [40.602 ms, 41.683 ms, 43.009 ms] | [62.782 ms, 64.308 ms, 65.802 ms] |
| 32 | [17.586 ms, 20.582 ms, 24.719 ms] | [14.5 ms, 14.992 ms, 15.531 ms] | [28.417 ms, 30.991 ms, 35.53 ms] | [191.68 ms, 194.67 ms, 197.95 ms] | [83.361 ms, 84.905 ms, 86.312 ms] | [125.06 ms, 128.46 ms, 133.26 ms] |
| 64 | [28.223 ms, 28.962 ms, 29.736 ms] | [24.68 ms, 25.428 ms, 26.204 ms] | [52.411 ms, 53.609 ms, 54.829 ms] | [362.23 ms, 366.11 ms, 369.9 ms] | [155.92 ms, 161.41 ms, 167.32 ms] | [234.98 ms, 242.88 ms, 250.97 ms] |
| 128 | [49.815 ms, 52.298 ms, 55.633 ms] | [44.431 ms, 46.651 ms, 49.73 ms] | [95.166 ms, 96.874 ms, 99.05 ms] | [739.98 ms, 748.01 ms, 755.64 ms] | [315.67 ms, 329.21 ms, 345.23 ms] | [476.69 ms, 484.98 ms, 493.26 ms] |
| 256 | [98.118 ms, 104.19 ms, 111.07 ms] | [85.759 ms, 88.479 ms, 91.91 ms] | [189.93 ms, 194.29 ms, 199.07 ms] | [1.4819 s, 1.514 s, 1.5533 s] | [830.33 ms, 896.38 ms, 962.93 ms] | [1.3273 s, 1.3717 s, 1.4169 s] |
| 512 | [233.25 ms, 242.1 ms, 252.67 ms] | [172.81 ms, 186.42 ms, 203.2 ms] | [447.91 ms, 499.13 ms, 568.98 ms] | [3.3497 s, 3.6689 s, 4.011 s] | [1.4856 s, 1.6523 s, 1.8235 s] | [2.0119 s, 2.0627 s, 2.1197 s] |
| 1024 | [348.62 ms, 369.31 ms, 399.08 ms] | [288.81 ms, 293.31 ms, 297.56 ms] | [746.41 ms, 762.17 ms, 776.82 ms] | [5.6866 s, 5.7137 s, 5.7411 s] | [2.4302 s, 2.4781 s, 2.5211 s] | [3.8245 s, 4.0096 s, 4.2137 s] |
| 2048 | [826.31 ms, 851.39 ms, 874.85 ms] | [597.25 ms, 605.91 ms, 614.38 ms] | [2.0671 s, 2.2936 s, 2.4947 s] | [11.48 s, 11.558 s, 11.648 s] | [4.6955 s, 4.7764 s, 4.9055 s] | [6.7178 s, 6.9567 s, 7.2805 s] |

## Relative to `outer-current` (midpoint ratios)
Ratios computed as `(variant_mid / outer-current_mid)` after converting each midpoint to ms.

| N | outer-uni-skip | outer-streaming | outer-round-batched | outer-baseline | outer-naive |
|---|---|---|---|---|---|
| 8 | 0.94× | 1.66× | 8.29× | 3.81× | 5.84× |
| 16 | 0.87× | 1.58× | 9.04× | 3.92× | 6.05× |
| 32 | 0.73× | 1.51× | 9.46× | 4.13× | 6.24× |
| 64 | 0.88× | 1.85× | 12.64× | 5.57× | 8.39× |
| 128 | 0.89× | 1.85× | 14.30× | 6.29× | 9.27× |
| 256 | 0.85× | 1.86× | 14.53× | 8.60× | 13.17× |
| 512 | 0.77× | 2.06× | 15.15× | 6.82× | 8.52× |
| 1024 | 0.79× | 2.06× | 15.47× | 6.71× | 10.86× |
| 2048 | 0.71× | 2.69× | 13.58× | 5.61× | 8.17× |

## Relative to `outer-uni-skip` (midpoint ratios)
Ratios computed as `(variant_mid / outer-uni-skip_mid)` after converting each midpoint to ms.

| N | outer-current | outer-streaming | outer-round-batched | outer-baseline | outer-naive |
|---|---|---|---|---|---|
| 8 | 1.06× | 1.76× | 8.82× | 4.05× | 6.21× |
| 16 | 1.15× | 1.82× | 10.43× | 4.52× | 6.98× |
| 32 | 1.37× | 2.07× | 12.98× | 5.66× | 8.57× |
| 64 | 1.14× | 2.11× | 14.40× | 6.35× | 9.55× |
| 128 | 1.12× | 2.08× | 16.03× | 7.06× | 10.40× |
| 256 | 1.18× | 2.20× | 17.11× | 10.13× | 15.50× |
| 512 | 1.30× | 2.68× | 19.68× | 8.86× | 11.06× |
| 1024 | 1.26× | 2.60× | 19.48× | 8.45× | 13.67× |
| 2048 | 1.41× | 3.79× | 19.08× | 7.88× | 11.48× |

## `outer-uni-skip` speedup vs `outer-current` (midpoints)
Computed as `(outer-current_mid / outer-uni-skip_mid)` (converted to ms). >1 means `outer-uni-skip` is faster.

| N | speedup |
|---|---|
| 8 | 1.063× |
| 16 | 1.154× |
| 32 | 1.373× |
| 64 | 1.139× |
| 128 | 1.121× |
| 256 | 1.178× |
| 512 | 1.299× |
| 1024 | 1.259× |
| 2048 | 1.405× |

## Scaling (midpoint per doubling)
For each variant: `mid(N*2)/mid(N)` (midpoints converted to ms).

- **outer-current**: 16:1.61×, 32:1.94×, 64:1.41×, 128:1.81×, 256:1.99×, 512:2.32×, 1024:1.53×, 2048:2.31× (geo-mean 1.84×)
- **outer-uni-skip**: 16:1.49×, 32:1.63×, 64:1.70×, 128:1.83×, 256:1.90×, 512:2.11×, 1024:1.57×, 2048:2.07× (geo-mean 1.77×)
- **outer-streaming**: 16:1.54×, 32:1.84×, 64:1.73×, 128:1.81×, 256:2.01×, 512:2.57×, 1024:1.53×, 2048:3.01× (geo-mean 1.95×)
- **outer-round-batched**: 16:1.76×, 32:2.03×, 64:1.88×, 128:2.04×, 256:2.02×, 512:2.42×, 1024:1.56×, 2048:2.02× (geo-mean 1.95×)
- **outer-baseline**: 16:1.66×, 32:2.04×, 64:1.90×, 128:2.04×, 256:2.72×, 512:1.84×, 1024:1.50×, 2048:1.93× (geo-mean 1.93×)
- **outer-naive**: 16:1.67×, 32:2.00×, 64:1.89×, 128:2.00×, 256:2.83×, 512:1.50×, 1024:1.94×, 2048:1.74× (geo-mean 1.91×)

## Notes
- Many benchmarks auto-extended beyond the requested 1s measurement time to reach 10 samples; treat this as a quick run for shape/comparison, not a final tuned measurement.
- Criterion can mix units within the bracket; this report preserves units and converts per-value for ratios.

# Spartan outer sumcheck benchmark results (sha2-chain) — run with `outer-uni-skip`

Source: `/Users/quang.dao/.cursor/projects/Users-quang-dao-Documents-SNARKs-jolt-spartan/agent-tools/1d46f132-7838-4bf3-8b82-d6d33504b7ae.txt`

## Extracted timings
All times are Criterion-estimated \([low, mid, high]\). Units are preserved per value.

| N | outer-current | outer-uni-skip | outer-streaming | outer-round-batched | outer-baseline | outer-naive |
|---|---|---|---|---|---|---|
| 8 | [6.3808 ms, 6.5957 ms, 6.8447 ms] | [6.0203 ms, 6.2032 ms, 6.4074 ms] | [10.271 ms, 10.924 ms, 11.549 ms] | [53.438 ms, 54.685 ms, 55.987 ms] | [24.565 ms, 25.102 ms, 25.672 ms] | [35.054 ms, 38.513 ms, 43.12 ms] |
| 16 | [10.303 ms, 10.629 ms, 10.974 ms] | [8.9795 ms, 9.2142 ms, 9.4728 ms] | [16.121 ms, 16.803 ms, 17.472 ms] | [92.531 ms, 96.109 ms, 99.836 ms] | [40.602 ms, 41.683 ms, 43.009 ms] | [62.782 ms, 64.308 ms, 65.802 ms] |
| 32 | [17.586 ms, 20.582 ms, 24.719 ms] | [14.5 ms, 14.992 ms, 15.531 ms] | [28.417 ms, 30.991 ms, 35.53 ms] | [191.68 ms, 194.67 ms, 197.95 ms] | [83.361 ms, 84.905 ms, 86.312 ms] | [125.06 ms, 128.46 ms, 133.26 ms] |
| 64 | [28.223 ms, 28.962 ms, 29.736 ms] | [24.68 ms, 25.428 ms, 26.204 ms] | [52.411 ms, 53.609 ms, 54.829 ms] | [362.23 ms, 366.11 ms, 369.9 ms] | [155.92 ms, 161.41 ms, 167.32 ms] | [234.98 ms, 242.88 ms, 250.97 ms] |
| 128 | [49.815 ms, 52.298 ms, 55.633 ms] | [44.431 ms, 46.651 ms, 49.73 ms] | [95.166 ms, 96.874 ms, 99.05 ms] | [739.98 ms, 748.01 ms, 755.64 ms] | [315.67 ms, 329.21 ms, 345.23 ms] | [476.69 ms, 484.98 ms, 493.26 ms] |
| 256 | [98.118 ms, 104.19 ms, 111.07 ms] | [85.759 ms, 88.479 ms, 91.91 ms] | [189.93 ms, 194.29 ms, 199.07 ms] | [1.4819 s, 1.514 s, 1.5533 s] | [830.33 ms, 896.38 ms, 962.93 ms] | [1.3273 s, 1.3717 s, 1.4169 s] |
| 512 | [233.25 ms, 242.1 ms, 252.67 ms] | [172.81 ms, 186.42 ms, 203.2 ms] | [447.91 ms, 499.13 ms, 568.98 ms] | [3.3497 s, 3.6689 s, 4.011 s] | [1.4856 s, 1.6523 s, 1.8235 s] | [2.0119 s, 2.0627 s, 2.1197 s] |
| 1024 | [348.62 ms, 369.31 ms, 399.08 ms] | [288.81 ms, 293.31 ms, 297.56 ms] | [746.41 ms, 762.17 ms, 776.82 ms] | [5.6866 s, 5.7137 s, 5.7411 s] | [2.4302 s, 2.4781 s, 2.5211 s] | [3.8245 s, 4.0096 s, 4.2137 s] |
| 2048 | [826.31 ms, 851.39 ms, 874.85 ms] | [597.25 ms, 605.91 ms, 614.38 ms] | [2.0671 s, 2.2936 s, 2.4947 s] | [11.48 s, 11.558 s, 11.648 s] | [4.6955 s, 4.7764 s, 4.9055 s] | [6.7178 s, 6.9567 s, 7.2805 s] |

## Relative to `outer-current` (midpoint ratios)
Ratios computed as `(variant_mid / outer-current_mid)` after converting each midpoint to ms.

| N | outer-uni-skip | outer-streaming | outer-round-batched | outer-baseline | outer-naive |
|---|---|---|---|---|---|
| 8 | 0.94× | 1.66× | 8.29× | 3.81× | 5.84× |
| 16 | 0.87× | 1.58× | 9.04× | 3.92× | 6.05× |
| 32 | 0.73× | 1.51× | 9.46× | 4.13× | 6.24× |
| 64 | 0.88× | 1.85× | 12.64× | 5.57× | 8.39× |
| 128 | 0.89× | 1.85× | 14.30× | 6.29× | 9.27× |
| 256 | 0.85× | 1.86× | 14.53× | 8.60× | 13.17× |
| 512 | 0.77× | 2.06× | 15.15× | 6.82× | 8.52× |
| 1024 | 0.79× | 2.06× | 15.47× | 6.71× | 10.86× |
| 2048 | 0.71× | 2.69× | 13.58× | 5.61× | 8.17× |

## Relative to `outer-uni-skip` (midpoint ratios)
Ratios computed as `(variant_mid / outer-uni-skip_mid)` after converting each midpoint to ms.

| N | outer-current | outer-streaming | outer-round-batched | outer-baseline | outer-naive |
|---|---|---|---|---|---|
| 8 | 1.06× | 1.76× | 8.82× | 4.05× | 6.21× |
| 16 | 1.15× | 1.82× | 10.43× | 4.52× | 6.98× |
| 32 | 1.37× | 2.07× | 12.98× | 5.66× | 8.57× |
| 64 | 1.14× | 2.11× | 14.40× | 6.35× | 9.55× |
| 128 | 1.12× | 2.08× | 16.03× | 7.06× | 10.40× |
| 256 | 1.18× | 2.20× | 17.11× | 10.13× | 15.50× |
| 512 | 1.30× | 2.68× | 19.68× | 8.86× | 11.06× |
| 1024 | 1.26× | 2.60× | 19.48× | 8.45× | 13.67× |
| 2048 | 1.41× | 3.79× | 19.08× | 7.88× | 11.48× |

## `outer-uni-skip` speedup vs `outer-current` (midpoints)
Computed as `(outer-current_mid / outer-uni-skip_mid)` (converted to ms). >1 means `outer-uni-skip` is faster.

| N | speedup |
|---|---|
| 8 | 1.063× |
| 16 | 1.154× |
| 32 | 1.373× |
| 64 | 1.139× |
| 128 | 1.121× |
| 256 | 1.178× |
| 512 | 1.299× |
| 1024 | 1.259× |
| 2048 | 1.405× |

## Scaling (midpoint per doubling)
For each variant: `mid(N*2)/mid(N)` (midpoints converted to ms).

- **outer-current**: 16:1.61×, 32:1.94×, 64:1.41×, 128:1.81×, 256:1.99×, 512:2.32×, 1024:1.53×, 2048:2.31× (geo-mean 1.84×)
- **outer-uni-skip**: 16:1.49×, 32:1.63×, 64:1.70×, 128:1.83×, 256:1.90×, 512:2.11×, 1024:1.57×, 2048:2.07× (geo-mean 1.77×)
- **outer-streaming**: 16:1.54×, 32:1.84×, 64:1.73×, 128:1.81×, 256:2.01×, 512:2.57×, 1024:1.53×, 2048:3.01× (geo-mean 1.95×)
- **outer-round-batched**: 16:1.76×, 32:2.03×, 64:1.88×, 128:2.04×, 256:2.02×, 512:2.42×, 1024:1.56×, 2048:2.02× (geo-mean 1.95×)
- **outer-baseline**: 16:1.66×, 32:2.04×, 64:1.90×, 128:2.04×, 256:2.72×, 512:1.84×, 1024:1.50×, 2048:1.93× (geo-mean 1.93×)
- **outer-naive**: 16:1.67×, 32:2.00×, 64:1.89×, 128:2.00×, 256:2.83×, 512:1.50×, 1024:1.94×, 2048:1.74× (geo-mean 1.91×)

## Notes
- Many benchmarks auto-extended beyond the requested 1s measurement time to reach 10 samples; treat this as a quick run for shape/comparison, not a final tuned measurement.
- Criterion can mix units within the bracket; this report preserves units and converts per-value for ratios.

