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
