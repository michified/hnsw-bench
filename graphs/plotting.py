# results

# naive k=1
# Queries per second @ top-1: 4.91823

# naive k=10
# Queries per second @ top-10: 4.63879

# naive k=25
# Queries per second @ top-25: 5.21334

# naive k=100
# Queries per second @ top-100: 5.09408

# HNSW k=1
# Build time: 72.5765 seconds
# Average query time: 711.972 microseconds
# Queries per second: 1404.55
# Average recall: 64.46%

# HNSW k=10
# Build time: 65.558 seconds
# Average query time: 2974.78 microseconds
# Queries per second: 336.159
# Average recall: 96.592%

# HNSW k=25
# Build time: 63.0662 seconds
# Average query time: 5509.83 microseconds
# Queries per second: 181.494
# Average recall: 98.5948%

# HNSW k=100
# Build time: 54.4366 seconds
# Average query time: 12147.7 microseconds
# Queries per second: 82.3199
# Average recall: 99.6371%

# annoy k=1
# Annoy build time: 3.8797 seconds
# Average time per query: 467.5138 microseconds
# Queries per second: 2138.9744
# Average recall@1: 89.2200%

# annoy k=10
# Annoy build time: 3.5853 seconds
# Average time per query: 502.7250 microseconds
# Queries per second: 1989.1590
# Average recall@10: 83.1810%

# annoy k=25
# Annoy build time: 3.5537 seconds
# Average time per query: 511.2578 microseconds
# Queries per second: 1955.9604
# Average recall@25: 78.8820%

# annoy k=100
# Annoy build time: 3.6906 seconds
# Average time per query: 718.1261 microseconds
# Queries per second: 1392.5132
# Average recall@100: 81.7116%

# HNSWLib k=1
# HNSW build time: 2.8288 seconds
# Average time per query: 230.9597 microseconds
# Queries per second: 4329.7597
# Average recall@100: 99.1232%

# HNSWLib k=10
# HNSW build time: 2.9218 seconds
# Average time per query: 232.3732 microseconds
# Queries per second: 4303.4224
# Average recall@100: 99.1433%

# HNSWLib k=25
# HNSW build time: 2.8914 seconds
# Average time per query: 321.5792 microseconds
# Queries per second: 3109.6541
# Average recall@100: 99.1388%

# HNSWLib k=100
# HNSW build time: 2.9614 seconds
# Average time per query: 295.6289 microseconds
# Queries per second: 3382.6189
# Average recall@100: 99.1452%

# plotting
import matplotlib.pyplot as plt
import pandas as pd

# Set matplotlib style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 10

# Colors for different methods
COLORS = {
    'Naive': '#1f77b4',
    'HNSW': '#ff7f0e',
    'Annoy': '#2ca02c',
    'HNSWLib': '#d62728'
}

# Create data for the table
data = {
    'Method': ['Naive']*4 + ['HNSW']*4 + ['Annoy']*4 + ['HNSWLib']*4,
    'k': [1, 10, 25, 100] * 4,
    'Build Time (s)': [
        0, 0, 0, 0,  # Naive (no build time)
        72.5765, 65.558, 63.0662, 54.4366,  # HNSW
        3.8797, 3.5853, 3.5537, 3.6906,  # Annoy
        2.8288, 2.9218, 2.8914, 2.9614  # HNSWLib
    ],
    'Query Time (µs)': [
        1000000/4.91823, 1000000/4.63879, 1000000/5.21334, 1000000/5.09408,  # Naive (converted from QPS)
        711.972, 2974.78, 5509.83, 12147.7,  # HNSW
        467.5138, 502.7250, 511.2578, 718.1261,  # Annoy
        230.9597, 232.3732, 321.5792, 295.6289  # HNSWLib
    ],
    'QPS': [
        4.91823, 4.63879, 5.21334, 5.09408,  # Naive
        1404.55, 336.159, 181.494, 82.3199,  # HNSW
        2138.9744, 1989.1590, 1955.9604, 1392.5132,  # Annoy
        4329.7597, 4303.4224, 3109.6541, 3382.6189  # HNSWLib
    ],
    'Recall (%)': [
        100, 100, 100, 100,  # Naive (exact search)
        64.46, 96.592, 98.5948, 99.6371,  # HNSW
        89.22, 83.181, 78.882, 81.7116,  # Annoy
        99.1232, 99.1433, 99.1388, 99.1452  # HNSWLib
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Print formatted table
print("\nPerformance Comparison Table:")
print(df.to_string(index=False))

# Save table to CSV
df.to_csv('performance_comparison.csv', index=False)

# Create figure with multiple subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Recall vs k
for method in ['HNSW', 'Annoy', 'HNSWLib']:
    method_data = df[df['Method'] == method]
    ax1.plot(method_data['k'], method_data['Recall (%)'], 'o-', 
             label=method, color=COLORS[method], linewidth=2, markersize=8)

ax1.set_xscale('log')
ax1.set_xlabel('k (Number of neighbors)', fontsize=12)
ax1.set_ylabel('Recall (%)', fontsize=12)
ax1.set_title('Recall vs k for Different Methods', fontsize=14)
ax1.grid(True, which="both", ls="-", alpha=0.2)
ax1.legend(fontsize=10)

# Plot 2: Query Performance vs k
for method in ['Naive', 'HNSW', 'Annoy', 'HNSWLib']:
    method_data = df[df['Method'] == method]
    ax2.plot(method_data['k'], method_data['QPS'], 'o-', 
             label=method, color=COLORS[method], linewidth=2, markersize=8)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('k (Number of neighbors)', fontsize=12)
ax2.set_ylabel('Queries per Second (log scale)', fontsize=12)
ax2.set_title('Query Performance vs k for Different Methods', fontsize=14)
ax2.grid(True, which="both", ls="-", alpha=0.2)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create an additional plot for build times
plt.figure(figsize=(12, 6))
build_times = df[df['Method'] != 'Naive'].groupby('Method')['Build Time (s)'].mean()

# Create bar plot with custom colors
ax = plt.gca()
bars = plt.bar(range(len(build_times)), build_times.values)
plt.xticks(range(len(build_times)), build_times.index, rotation=0)


for bar, method in zip(bars, build_times.index):
    bar.set_color(COLORS[method])
    bar.set_alpha(0.8)

plt.title('Average Build Time Comparison', fontsize=14)
plt.xlabel('Method', fontsize=12)
plt.ylabel('Build Time (seconds)', fontsize=12)
plt.grid(True, axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig('build_times_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
