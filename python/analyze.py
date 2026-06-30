import json 
import sys
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
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

df["median_ms"] = (
    df.groupby("rank")["gpu_ms"]
      .transform("median")
)

df["delta_ms"] = df["gpu_ms"] - df["median_ms"]

for rank, g in df.groupby("rank"):
    print(f"Rank {rank}")
    print(f"  p50: {g['gpu_ms'].quantile(0.50):.3f} ms")
    print(f"  p95: {g['gpu_ms'].quantile(0.95):.3f} ms")
    print(f"  p99: {g['gpu_ms'].quantile(0.99):.3f} ms")
    print(f"  max: {g['gpu_ms'].max():.3f} ms")
    print(f"  max_delta: {g['delta_ms'].max():.3f} ms")

pivot = df.pivot(index="iter", columns="rank", values="gpu_ms")
pivot["max_ms"] = pivot.max(axis=1)
pivot["min_ms"] = pivot.min(axis=1)
pivot["ratio"] = pivot["max_ms"] / pivot["min_ms"]

spikes = pivot[pivot["ratio"] > 1.5]

print("Latency spikes")
print("--------------")
print(f"count: {len(spikes)}")

rank_cols = [c for c in pivot.columns if c not in ["max_ms", "min_ms", "ratio"]]

likely_delayed_counts = {}

"""
impacted_rank is the rank with the biggest observed latency. likely_delayed_rank is the rank with the smallest observed latency during spike iterations.
"""

for iter_id, row in spikes.head(10).iterrows():
    impacted_rank = row[rank_cols].idxmax()
    likely_delayed_rank = row[rank_cols].idxmin()

    likely_delayed_counts[likely_delayed_rank] = (
        likely_delayed_counts.get(likely_delayed_rank, 0) + 1
    )

    print(
        f"iter {iter_id}: impacted rank {impacted_rank} observed {row['max_ms']:.3f} ms "
        f"({row['ratio']:.2f}x fastest rank); "
        f"likely delayed rank {likely_delayed_rank}"
    )

if likely_delayed_counts:
    likely_delayed_rank = max(likely_delayed_counts, key=likely_delayed_counts.get)

    print()
    print("Likely delayed rank")
    print("-------------------")
    print(
        f"rank {likely_delayed_rank} "
        f"({likely_delayed_counts[likely_delayed_rank]}/{len(spikes.head(10))} inspected spikes)"
    )

Path("plots").mkdir(exist_ok=True)

# 1. Latency over time
fig = px.line(
    df,
    x="iter",
    y="gpu_ms",
    color=df["rank"].astype(str),
    markers=True,
    title="NCCL AllReduce latency by rank",
    labels={
        "iter": "Iteration",
        "gpu_ms": "GPU latency (ms)",
        "color": "Rank",
    },
)

fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    legend_title_text="Rank",
)

for spike_iter in spikes.index:
    fig.add_vrect(
        x0=spike_iter - 0.5,
        x1=spike_iter + 0.5,
        fillcolor="red",
        opacity=0.08,
        line_width=0,
    )

fig.write_html("plots/latency_by_rank.html")


# 2. Latency distribution by rank
fig = px.box(
    df,
    x=df["rank"].astype(str),
    y="gpu_ms",
    points="outliers",
    title="NCCL latency distribution by rank",
    labels={
        "x": "Rank",
        "gpu_ms": "GPU latency (ms)",
    },
)

fig.update_layout(
    template="plotly_white",
)

fig.write_html("plots/latency_distribution.html")


# 3. Heatmap: rank x iteration
heatmap = df.pivot(
    index="rank",
    columns="iter",
    values="gpu_ms",
)

fig = px.imshow(
    heatmap,
    aspect="auto",
    title="NCCL latency heatmap",
    labels={
        "x": "Iteration",
        "y": "Rank",
        "color": "GPU latency (ms)",
    },
)

fig.update_layout(
    template="plotly_white",
)

fig.write_html("plots/latency_heatmap.html")

print()
print("Wrote plots:")
print("  plots/latency_by_rank.html")
print("  plots/latency_distribution.html")
print("  plots/latency_heatmap.html")

