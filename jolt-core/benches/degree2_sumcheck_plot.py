import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> None:
    # Sumcheck variable counts (ℓ) for which we have single-thread medians
    vars_ = [14, 16, 18, 20, 22, 24, 26]

    # Single-threaded medians in milliseconds
    # Optimized MontU128 challenge (no challenge-254-bit feature)
    plain_opt = [1.0828, 5.2203, 20.854, 83.872, 338.04, 1391.8, 5969.7]
    mul_opt = [0.85156, 3.9697, 15.439, 62.790, 251.96, 1031.3, 4098.7]

    # Unoptimized 254-bit challenge (with --features challenge-254-bit)
    plain_unopt = [1.2039, 5.5572, 21.740, 86.571, 344.40, 1386.1, 5951.3]
    mul_unopt = [0.95411, 4.1842, 16.123, 66.034, 259.47, 1064.2, 4464.4]

    # Baseline: plain multiplication with the unoptimized 254-bit challenge
    baseline = plain_unopt

    speed_baseline = [1.0] * len(vars_)
    speed_opt_challenge = [b / t for b, t in zip(baseline, plain_opt)]
    speed_delayed = [b / t for b, t in zip(baseline, mul_unopt)]
    speed_both = [b / t for b, t in zip(baseline, mul_opt)]

    plt.style.use("default")
    plt.figure(figsize=(9, 4.3))
    ax = plt.gca()

    ax.plot(
        vars_,
        speed_baseline,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label="baseline (unopt challenge, no delayed reduction)",
    )
    ax.plot(
        vars_,
        speed_opt_challenge,
        marker="s",
        color="#1f77b4",
        linewidth=2,
        label="optimized challenge only",
    )
    ax.plot(
        vars_,
        speed_delayed,
        marker="o",
        color="#ff7f0e",
        linewidth=2,
        label="delayed reduction only",
    )
    ax.plot(
        vars_,
        speed_both,
        marker="^",
        color="#2ca02c",
        linewidth=2,
        label="both optimizations",
    )

    ax.set_ylim(0.95, 1.55)
    ax.set_xticks(vars_)
    # Enlarge tick label fonts by ~1.5x and keep axes unlabeled (no text on x/y).
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, linestyle="--", linewidth=0.4)
    ax.legend(loc="upper right")

    plt.tight_layout()
    output_path = Path("jolt-core/benches/degree2_sumcheck_speedup.png")
    plt.savefig(output_path, dpi=220)


if __name__ == "__main__":
    main()


