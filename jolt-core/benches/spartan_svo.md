**Benchmark Results Summary (sha2-chain-guest):**

*   **8 Iterations:**
    *   Original: ~280 ms
    *   Gruen: ~193 ms (1.45x speedup over Original)
    *   Gruen + 3 SVO: ~119 ms (1.62x speedup over Gruen, 2.35x speedup over Original)

*   **16 Iterations:**
    *   Original: ~546 ms
    *   Gruen: ~370 ms (1.48x speedup over Original)
    *   Gruen + 3 SVO: ~221 ms (1.67x speedup over Gruen, 2.47x speedup over Original)

*   **32 Iterations:**
    *   Original: ~1.08 s
    *   Gruen: ~713 ms (1.51x speedup over Original)
    *   Gruen + 3 SVO: ~453 ms (1.57x speedup over Gruen, 2.38x speedup over Original)

*   **64 Iterations:**
    *   Original: ~2.24 s
    *   Gruen: ~1.46 s (1.53x speedup over Original)
    *   Gruen + 3 SVO: ~1.01 s (1.45x speedup over Gruen, 2.22x speedup over Original)

*   **128 Iterations:**
    *   Original: ~5.95 s
    *   Gruen: ~2.86 s (2.08x speedup over Original)
    *   Gruen + 3 SVO: ~1.85 s (1.55x speedup over Gruen, 3.22x speedup over Original)

*   **256 Iterations:**
    *   Original: ~10.84 s
    *   Gruen: ~6.94 s (1.56x speedup over Original)
    *   Gruen + 3 SVO: ~3.70 s (1.88x speedup over Gruen, 2.93x speedup over Original)

*   **512 Iterations:**
    *   Original: ~33.73 s
    *   Gruen: ~24.79 s (1.36x speedup over Original)
    *   Gruen + 3 SVO: ~7.35 s (3.37x speedup over Gruen, 4.59x speedup over Original)

*   **1024 Iterations:**
    *   Original: ~278.39 s
    *   Gruen: ~237.58 s (1.17x speedup over Original)
    *   Gruen + 3 SVO: ~16.69 s (14.2x speedup over Gruen, 16.7x speedup over Original)

*   **2048 Iterations:**
    *   Original: N/A (OOM / Timeout)
    *   Gruen: N/A (OOM / Timeout)
    *   Gruen + 3 SVO: ~48.20 s (Comparison N/A as baseline data unavailable for 2048 iters)

**Key Observations:**
*   "Gruen + 3 SVO" consistently provides significant speedups over both "Gruen" alone and the "Original" implementation, with speedups ranging from 2.2x to 16.7x compared to Original.
*   "Gruen" alone provides modest but consistent improvements over the "Original" implementation (1.17x to 2.08x speedup).
*   The speedup advantage of "Gruen + 3 SVO" becomes dramatically more pronounced at higher iteration counts, achieving 16.7x speedup at 1024 iterations.
*   The "Gruen + 3 SVO" method is the only one able to complete 2048 iterations within reasonable time/memory constraints. 