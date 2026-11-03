"""Learning-speed figures for the presentation (docs/presentation/images/).

Reads every run under logs/rsl_rl/ and reuses the helpers of logs/learning_speed.py (same smoothing, same
"final value", same 90% rule), so the figures match the table printed by that script.

Writes:
  images/learning_speed_vs_params.png   Experiment A: iterations to 90% vs actor parameters
  images/learning_speed_vs_depth.png    Experiment B: iterations to 90% vs number of hidden layers
  images/learning_speed_curves_expA.png smoothed reward curves with the 90% crossing, Experiment A
  images/learning_speed_curves_expB.png same, Experiment B

Usage (repo root, .venv active):  python docs/presentation/make_learning_speed_plots.py
"""

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "logs"))
import learning_speed as ls  # noqa: E402  (reuses load_run, smooth, first_reach)

OUT = os.path.join(HERE, "images")
WINDOW = 20
EXP_A = ["extra_small", "small", "medium", "baseline", "large"]  # ordered by parameter count
EXP_B = ["deep_1", "deep_2", "baseline", "deep_5"]  # ordered by number of hidden layers
PREFIX = "quadrustand_spot_flat_"

# Colours (light mode, validated with the dataviz validator: categorical slots 1-4, surface #fcfcfb)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order, never cycled


def actor_params(hidden_dims, n_in=48, n_out=12):
    sizes = [n_in, *hidden_dims, n_out]
    return sum(i * o + o for i, o in zip(sizes[:-1], sizes[1:]))


def collect():
    runs = []
    for d in sorted(glob.glob(os.path.join(ls.LOGS_ROOT, "*", "*"))):
        if not os.path.isdir(d):
            continue
        loaded = ls.load_run(d)
        if loaded is None:
            continue
        steps, values, seed = loaded
        with open(os.path.join(d, "params", "agent.yaml")) as f:
            dims = yaml.safe_load(f)["actor"]["hidden_dims"]
        n_tail = max(1, int(round(0.05 * len(values))))
        runs.append(
            {
                "variant": os.path.basename(os.path.dirname(d)).replace(PREFIX, ""),
                "seed": seed,
                "steps": steps,
                "smooth": ls.smooth(values, WINDOW),
                "final": float(values[-n_tail:].mean()),
                "dims": dims,
                "params": actor_params(dims),
            }
        )
    baseline_final = float(np.mean([r["final"] for r in runs if r["variant"] == "baseline"]))
    for r in runs:
        r["own_90"] = ls.first_reach(r["steps"], r["smooth"], 0.9 * r["final"])
        r["base_90"] = ls.first_reach(r["steps"], r["smooth"], 0.9 * baseline_final)
    return runs, baseline_final


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def speed_figure(runs, order, x_of, tick_label, xlabel, title, fname, logx=False):
    """Iterations to reach 90%, one point per variant (seed 1), baseline seeds and noise band shown."""
    seed1 = {r["variant"]: r for r in runs if r["seed"] == 1}
    base_runs = [r for r in runs if r["variant"] == "baseline"]
    metrics = [("own_90", "to 90% of the run's own final reward"), ("base_90", "to 90% of the baseline's final reward")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=SURFACE)
    for ax, (key, panel_title) in zip(axes, metrics):
        style(ax)
        base_vals = [r[key] for r in base_runs if r[key] is not None]
        lo, hi = min(base_vals), max(base_vals)
        ax.axhspan(lo, hi, color=AXIS, alpha=0.4, linewidth=0, zorder=1)
        xs = [x_of(seed1[v]) for v in order]
        ys = [seed1[v][key] for v in order]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        ax.plot(*zip(*pts), color=BLUE, linewidth=2, marker="o", markersize=9, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=3, label="seed 1 (one run per variant)")  # fmt: skip
        extra = [r for r in base_runs if r["seed"] != 1 and r[key] is not None]
        ax.plot([x_of(r) for r in extra], [r[key] for r in extra], linestyle="none", marker="o", markersize=9,
                markerfacecolor=SURFACE, markeredgecolor=INK2, markeredgewidth=2, zorder=3,
                label="baseline seeds 2 and 3")  # fmt: skip
        ax.plot([], [], color=AXIS, linewidth=8, alpha=0.4, label="baseline noise band (min–max of seeds 1–3)")
        for x, y, v in zip(xs, ys, order):
            if y is None:
                ax.annotate("never", (x, 0), xytext=(0, 8), textcoords="offset points", ha="center", color=INK)
            elif y > hi or y < lo:  # selective labels: only values outside the noise band
                ax.annotate(str(y), (x, y), xytext=(0, 12), textcoords="offset points", ha="center",
                            fontsize=11, color=INK)  # fmt: skip
        ax.set_xticks(xs)
        ax.set_xticklabels([tick_label(v) for v in order], fontsize=10)
        if logx:
            ax.set_xscale("log")
            ax.minorticks_off()
            ax.set_xticks(xs)
            ax.set_xticklabels([tick_label(v) for v in order], fontsize=10)
        ax.set_ylim(0, max(v for v in ys if v is not None) * 1.15)
        ax.set_xlabel(xlabel, color=INK2, fontsize=12)
        ax.set_ylabel("iterations (lower = faster)", color=INK2, fontsize=12)
        ax.set_title(panel_title, color=INK2, fontsize=12, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()  # one shared legend below the plots, off the data
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=11, labelcolor=INK2)
    fig.suptitle(title, color=INK, fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(os.path.join(OUT, fname), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def curves_figure(runs, order, baseline_final, title, fname):
    """Smoothed reward curves; marks where each variant crosses 90% of the baseline's final reward."""
    seed1 = {r["variant"]: r for r in runs if r["seed"] == 1}
    base_runs = [r for r in runs if r["variant"] == "baseline"]
    n = min(len(r["smooth"]) for r in base_runs)
    stack = np.stack([r["smooth"][:n] for r in base_runs])
    steps = base_runs[0]["steps"][:n]
    base_cross = [r["base_90"] for r in base_runs if r["base_90"] is not None]
    target = 0.9 * baseline_final

    fig, ax = plt.subplots(figsize=(12, 5.6), facecolor=SURFACE)
    style(ax)
    ax.xaxis.grid(False)
    ax.axvspan(min(base_cross), max(base_cross), color=AXIS, alpha=0.35, linewidth=0, zorder=1,
               label=f"baseline seeds cross the line between {min(base_cross)} and {max(base_cross)}")  # fmt: skip
    ax.fill_between(steps, stack.min(0), stack.max(0), color=AXIS, alpha=0.5, linewidth=0, zorder=2)
    ax.plot(steps, stack.mean(0), color=INK2, linewidth=2, zorder=3, label="baseline (mean of 3 seeds, band = min–max)")
    variants = [v for v in order if v != "baseline"]
    for color, v in zip(CATEGORICAL, variants):
        r = seed1[v]
        cross = r["base_90"]
        text = f"{v} (seed 1): crosses at {cross}" if cross is not None else f"{v} (seed 1): never crosses"
        ax.plot(r["steps"], r["smooth"], color=color, linewidth=2, zorder=4, label=text)
        if cross is not None:
            ax.plot([cross], [target], marker="o", markersize=10, color=color, markeredgecolor=SURFACE,
                    markeredgewidth=2, zorder=6)  # fmt: skip
    ax.axhline(target, color=INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=5,
               label=f"90% of the baseline's final reward ({target:.0f})")  # fmt: skip
    ax.set_xlim(0, steps[-1])
    ax.set_ylim(0, max(baseline_final, *(seed1[v]["final"] for v in variants)) * 1.12)
    ax.set_xlabel("training iteration", color=INK2, fontsize=12)
    ax.set_ylabel(f"mean reward (smoothed, {WINDOW}-iteration moving average)", color=INK2, fontsize=12)
    ax.legend(loc="lower right", frameon=False, fontsize=10.5, labelcolor=INK2)
    fig.suptitle(title, color=INK, fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, fname), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    runs, baseline_final = collect()
    print(f"baseline final reward (mean of seeds): {baseline_final:.3f}")
    for r in sorted(runs, key=lambda r: (r["variant"], r["seed"])):
        print(f"{r['variant']:12} seed {r['seed']}  params {r['params']:>7}  own_90 {r['own_90']}  base_90 {r['base_90']}")

    by_variant = {r["variant"]: r for r in runs if r["seed"] == 1}
    speed_figure(
        runs, EXP_A, lambda r: r["params"],
        lambda v: f"{v}\n{by_variant[v]['params'] / 1000:.0f}k",
        "actor parameters (log scale)",
        "Experiment A: learning speed vs network size", "learning_speed_vs_params.png", logx=True,
    )  # fmt: skip
    speed_figure(
        runs, EXP_B, lambda r: len(r["dims"]),
        lambda v: f"{v}\n{len(by_variant[v]['dims'])} layer(s)",
        "hidden layers (about 190k parameters each)",
        "Experiment B: learning speed vs depth at a fixed parameter budget", "learning_speed_vs_depth.png",
    )  # fmt: skip
    curves_figure(runs, EXP_A, baseline_final, "Experiment A: where each curve reaches 90% of the baseline's final reward",
                  "learning_speed_curves_expA.png")  # fmt: skip
    curves_figure(runs, EXP_B, baseline_final, "Experiment B: where each curve reaches 90% of the baseline's final reward",
                  "learning_speed_curves_expB.png")  # fmt: skip
    print("written to", OUT)


if __name__ == "__main__":
    main()
