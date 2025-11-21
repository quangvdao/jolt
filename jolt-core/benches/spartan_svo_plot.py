import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    # Iteration counts for the sha2-chain-guest benchmark
    iters = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048])

    # Times in milliseconds, taken from spartan_svo.md
    # "Gruen" → baseline
    baseline = np.array(
        [
            193,  # 8
            370,  # 16
            713,  # 32
            1460,  # 64 (1.46 s)
            2860,  # 128 (2.86 s)
            6940,  # 256 (6.94 s)
            24790,  # 512 (24.79 s)
            237580,  # 1024 (237.58 s)
            np.nan,  # 2048 (N/A / OOM)
        ]
    )

    # "Gruen + 3 SVO" → round batched
    round_batched = np.array(
        [
            119,  # 8
            221,  # 16
            453,  # 32
            1010,  # 64 (1.01 s)
            1850,  # 128 (1.85 s)
            3700,  # 256 (3.70 s)
            7350,  # 512 (7.35 s)
            16690,  # 1024 (16.69 s)
            48200,  # 2048 (48.20 s)
        ]
    )

    # Mask out the missing baseline point at 2048 iterations
    baseline_mask = ~np.isnan(baseline)

    plt.style.use("default")
    plt.figure(figsize=(10, 6), dpi=200)

    # Plot baseline (Gruen)
    plt.plot(
        iters[baseline_mask],
        baseline[baseline_mask],
        marker="^",
        markersize=7,
        linewidth=2.5,
        color="#1f77b4",
        label="Baseline",
    )

    # Plot round batched (Gruen + 3 SVO)
    plt.plot(
        iters,
        round_batched,
        marker="^",
        markersize=7,
        linewidth=2.5,
        color="#ff7f0e",
        label="Round Batched",
    )

    # Log-log scales similar to the reference figure
    plt.xscale("log", base=2)
    plt.yscale("log", base=10)

    plt.xticks(iters, iters, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)

    ax = plt.gca()

    # Only show the left and bottom axes lines (no top/right rectangle border)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = plt.legend(frameon=True, fontsize=12)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig("spartan_svo_runtime_baseline_vs_round_batched.png", dpi=300)


if __name__ == "__main__":
    main()


