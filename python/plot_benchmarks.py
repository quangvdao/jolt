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

# Set font to Times New Roman for academic look
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
plt.rcParams['font.size'] = 16  # Increased base font size

# Create the plot with more space for title
plt.figure(figsize=(10, 7))

# Plotting the data
plt.plot(original_iters, original_times_ms, marker='^', linestyle='-', color='blue', label='Original', linewidth=2, markersize=8)
plt.plot(gruen_iters, gruen_times_ms, marker='^', linestyle='-', color='teal', label='Gruen', linewidth=2, markersize=8)
plt.plot(gruen_svo_iters, gruen_svo_times_ms, marker='^', linestyle='-', color='darkorange', label='Gruen+SVO', linewidth=2, markersize=8)

# Setting scales
plt.xscale('log', base=2)
plt.yscale('log')

# Setting ticks - show all x-axis values with tighter spacing
plt.xticks(iterations, labels=[str(it) for it in iterations], fontsize=14)
plt.yticks(fontsize=14)

# Adjust x-axis to make spacing between iterations smaller
ax = plt.gca()
ax.set_xlim(6, 2500)  # Tighter x-axis limits

# Remove minor grid lines and only show major grid lines
plt.grid(True, which="major", ls="-", alpha=0.3, color='gray')

# Configure y-axis to only show major ticks with minor tick marks
ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs='auto', numticks=12))
ax.tick_params(which='minor', length=3, width=0.5, direction='in')  # Small tick marks pointing inward
ax.tick_params(which='major', length=6, width=1, direction='in')    # Larger tick marks for major

# Labels and Title with more spacing
plt.xlabel('Number of SHA-256 Iterations', fontsize=18, labelpad=15)
plt.ylabel('Runtime (ms)', fontsize=18, labelpad=15)
plt.title("Prover Runtime of Jolt's Spartan Sumcheck", fontsize=20, fontweight='bold', pad=25)

# Legend - positioned away from the bottom right corner
plt.legend(loc='center right', bbox_to_anchor=(0.95, 0.25), frameon=True, fancybox=True, shadow=True, fontsize=16)

# Adjust layout with more space at top
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Add more space at the top for title

# To save the figure:
plt.savefig('jolt_spartan_sumcheck_benchmark.png', bbox_inches='tight', dpi=300)

# To display the figure:
# plt.show()

print("Matplotlib script generated. You can run this in a Python environment with matplotlib and numpy.")
print("Uncomment plt.savefig() to save the image or plt.show() to display it.")