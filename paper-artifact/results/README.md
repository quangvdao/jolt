# Processed paper results

These CSV files contain the processed numerical values used in the camera-ready
version of “Speeding Up Sum-Check Proving” (ACM CCS 2026). They were transcribed
from the final paper tables and plotting inputs and checked against those
sources.

The files are not raw Criterion output, Perfetto traces, or independent reruns.
The historical raw outputs from the February 2026 measurement campaign are no
longer available. Use the commands in [`../../ARTIFACT.md`](../../ARTIFACT.md)
to generate new raw measurements from this revision.

Unless a file says otherwise, measurements used BN254 on a MacBook Pro with an
Apple M4 Max (16 CPU cores: 12 performance and 4 efficiency cores) and 64 GB
RAM. The `source` column distinguishes Criterion estimates from process-memory,
trace-derived, and end-to-end aggregates.

## Files

- `high_degree_product_us.csv`: baseline and optimized Criterion estimates for
  products of `d` multilinear polynomials in `v` variables.
- `degree2_sumcheck_ms.csv`: delayed-reduction microbenchmarks, both without and
  with an equality polynomial.
- `spartan_runtime_ms.csv`: Spartan outer-sum-check ablation and streaming
  Criterion estimates.
- `spartan_peak_rss_gb.csv`: process peak RSS for the linear-space Spartan
  ablation.
- `spartan_streaming_delta_peak_rss_gb.csv`: setup-normalized additional peak
  RSS for the linear-space and streaming variants.
- `ra_virtualization_ms.csv`: Shout RA-virtualization timings and prover shares.
- `e2e_sha2_chain.csv`: end-to-end proving aggregates and trace-derived shares.

Times are wall-clock measurements. `delta_peak_rss_gb` is the maximum process
RSS minus the RSS sampled immediately after trace/setup; it is not total process
memory. Blank cells mean that the paper did not report a value.
