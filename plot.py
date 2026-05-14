import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

try:
    df = pd.read_csv("benchmark_results.csv", comment="#")
except FileNotFoundError:
    print("Ошибка: Файл benchmark_results.csv не найден.")
    exit(1)

v_types = ["float3", "float4", "double3", "double4"]
n_vals = sorted(df["N"].unique())
bs_vals = sorted(df["BS"].unique())
colors = plt.cm.tab10


def plot_type(ax, vtype, metric, ylabel):
    """Все BS для одного типа: global — сплошная, shared — пунктир"""
    for i, bs in enumerate(bs_vals):
        for variant, style in [("global", "-"), ("shared", "--")]:
            sub = df[
                (df["type"] == vtype) & (df["variant"] == variant) & (df["BS"] == bs)
            ]
            if not sub.empty:
                ax.plot(
                    sub["N"],
                    sub[metric],
                    linestyle=style,
                    color=colors(i),
                    marker=".",
                    label=f"BS={bs} ({variant})",
                    linewidth=1.2,
                )
    ax.set_title(vtype, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(n) for n in n_vals], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("N")


# ─── Фигура 1: ms_per_pair ───
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
for i, vt in enumerate(v_types):
    plot_type(axes1[i // 2][i % 2], vt, "ms_per_pair", "мс/пара")
handles, labels = axes1[0][0].get_legend_handles_labels()
fig1.legend(
    handles,
    labels,
    loc="outside lower center",
    ncol=6,
    fontsize=8,
    title="BS (variant)",
)
fig1.suptitle("Задержка (ms/pair) — все BS", fontsize=14, fontweight="bold")
plt.savefig("plot_latency.png", dpi=150)
print("plot_latency.png")

# ─── Фигура 2: gints ───
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
for i, vt in enumerate(v_types):
    plot_type(axes2[i // 2][i % 2], vt, "gints", "GInt/s")
handles, labels = axes2[0][0].get_legend_handles_labels()
fig2.legend(
    handles,
    labels,
    loc="outside lower center",
    ncol=6,
    fontsize=8,
    title="BS (variant)",
)
fig2.suptitle("Производительность (GInt/s) — все BS", fontsize=14, fontweight="bold")
plt.savefig("plot_throughput.png", dpi=150)
print("plot_throughput.png")

# ─── Фигура 3: Сравнение типов (лучший BS) ───
best = df.loc[df.groupby(["type", "N", "variant"])["gints"].idxmax()].sort_values("N")

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
for vt in v_types:
    for variant, style, marker in [("global", "-", "s"), ("shared", "--", "o")]:
        sub = best[(best["type"] == vt) & (best["variant"] == variant)]
        axes3[0].plot(
            sub["N"],
            sub["ms_per_pair"],
            linestyle=style,
            marker=marker,
            label=f"{vt} ({variant})",
            linewidth=2,
        )
        axes3[1].plot(
            sub["N"],
            sub["gints"],
            linestyle=style,
            marker=marker,
            label=f"{vt} ({variant})",
            linewidth=2,
        )

for ax, title, ylabel in [
    (axes3[0], "Задержка (лучший BS)", "мс/пара"),
    (axes3[1], "Производительность (лучший BS)", "GInt/s"),
]:
    ax.set_title(title, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(n) for n in n_vals])
    ax.legend(title="Тип (variant)", fontsize=9)
    ax.set_xlabel("N")
    ax.set_ylabel(ylabel)

fig3.suptitle("CUDA N-Body: Global vs Shared Memory", fontsize=14, fontweight="bold")
plt.savefig("plot_comparison.png", dpi=150)
print("plot_comparison.png")
