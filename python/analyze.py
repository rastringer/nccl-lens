import json 
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

rows = []

for path in sys.argv[1:]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))

df = pd.DataFrame(rows)
df = df[df["warmup"] == False]

print("Run summary")
print("-----------")
print(f"rows: {len(df)}")
print(f"ranks: {sorted(df['rank'].unique())}")
print()

for rank, g in df.groupby("rank"):
    print(f"Rank {rank}")
    print(f"  p50: {g['gpu_ms'].quantile(0.50):.3f} ms")
    print(f"  p95: {g['gpu_ms'].quantile(0.95):.3f} ms")
    print(f"  p99: {g['gpu_ms'].quantile(0.99):.3f} ms")
    print(f"  max: {g['gpu_ms'].max():.3f} ms")
    print()

pivot = df.pivot(index="iter", columns="rank", values="gpu_ms")
pivot["max_ms"] = pivot.max(axis=1)
pivot["min_ms"] = pivot.min(axis=1)
pivot["ratio"] = pivot["max_ms"] / pivot["min_ms"]

spikes = pivot[pivot["ratio"] > 1.5]

print("Latency spikes")
print("--------------")
print(f"count: {len(spikes)}")

for iter_id, row in spikes.head(10).iterrows():
    slow_rank = row.drop(["max_ms", "min_ms", "ratio"]).idxmax()
    print(
        f"iter {iter_id}: rank {slow_rank} observed {row['max_ms']:.3f} ms "
        f"({row['ratio']:.2f}x fastest rank)"
    )                            


Path("plots").mkdir(exist_ok=True)

plt.figure()
for rank, g in df.groupby("rank"):
    plt.plot(g["iter"], g["gpu_ms"], label=f"rank {rank}")

plt.xlabel("Iteration")
plt.ylabel("GPU time (ms)")
plt.title("NCCL all-reduce latency by rank")
plt.legend()
plt.tight_layout()
plt.savefig("plots/latency_by_rank.png")

print()
print("Wrote plot: plots/latency_by_rank.png")
