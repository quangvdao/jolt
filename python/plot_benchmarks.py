import matplotlib.pyplot as plt
import numpy as np

# Data based on BENCHMARKS.md
iterations = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048])

# Runtimes in milliseconds
original_iters = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
original_times_ms = np.array([
    280,    # 8 iters
    546,    # 16 iters
    1080,   # 32 iters (1.08s)
    2240,   # 64 iters (2.24s)
    5950,   # 128 iters (5.95s)
    10840,  # 256 iters (10.84s)
    33730,  # 512 iters (33.73s)
    278390  # 1024 iters (278.39s)
])

gruen_iters = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
gruen_times_ms = np.array([
    193,    # 8 iters
    370,    # 16 iters
    713,    # 32 iters
    1460,   # 64 iters (1.46s)
    2860,   # 128 iters (2.86s)
    6940,   # 256 iters (6.94s)
    24790,  # 512 iters (24.79s)
    237580  # 1024 iters (237.58s)
])

gruen_svo_iters = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048])
gruen_svo_times_ms = np.array([
    119,    # 8 iters
    221,    # 16 iters
    453,    # 32 iters
    1010,   # 64 iters (1.01s)
    1850,   # 128 iters (1.85s)
    3700,   # 256 iters (3.70s)
    7350,   # 512 iters (7.35s)
    16690,  # 1024 iters (16.69s)
    48200   # 2048 iters (48.20s)
])

# Create the plot
plt.figure(figsize=(10, 7)) # Adjusted figure size for better legend placement

# Plotting the data
plt.plot(original_iters, original_times_ms, marker='^', linestyle='-', color='royalblue', label='Original')
plt.plot(gruen_iters, gruen_times_ms, marker='^', linestyle='-', color='teal', label='Gruen')
plt.plot(gruen_svo_iters, gruen_svo_times_ms, marker='^', linestyle='-', color='darkorange', label='Gruen+SVO')

# Setting scales
plt.xscale('log', base=2)
plt.yscale('log') # Default base 10

# Setting ticks
plt.xticks(iterations, labels=[str(it) for it in iterations])
# Matplotlib will auto-generate y-ticks, which is usually fine for log scale.
# If you need specific y-ticks, you can set them, e.g.:
# plt.yticks([100, 1000, 10000, 100000, 1000000], labels=['100', '1k', '10k', '100k', '1M'])


# Labels and Title
plt.xlabel('Number of SHA-256 Iterations')
plt.ylabel('Runtime (ms)')
plt.title("Prover Runtime of Jolt's Spartan Sumcheck")

# Legend - placed below the plot as in the example
# Adjust bbox_to_anchor and ncol as needed
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)

# Grid
plt.grid(True, which="both", ls="-", alpha=0.6) # Grid for both major and minor ticks

# Adjust layout to make space for the legend if it's outside
plt.tight_layout(rect=[0, 0.05, 1, 1]) # rect=[left, bottom, right, top]

# To save the figure:
plt.savefig('jolt_spartan_sumcheck_benchmark.png', bbox_inches='tight')

# To display the figure:
# plt.show()

print("Matplotlib script generated. You can run this in a Python environment with matplotlib and numpy.")
print("Uncomment plt.savefig() to save the image or plt.show() to display it.")