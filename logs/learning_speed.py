"""Learning speed of every training run found under logs/rsl_rl/ (BENCHMARK.md, section 6.1).

For each run the script reads ALL points of `Train/mean_reward` straight from the TensorBoard event
files (no thinning, unlike the TensorBoard web UI) and computes:

  final          mean of the raw reward over the last 5% of the iterations
  own 90%        first iteration where the smoothed reward reaches 90% of the run's own final value
  baseline 90%   first iteration where the smoothed reward reaches 90% of the baseline's final value
                 (baseline reference = mean `final` over all runs in the baseline folder)

"never" means the smoothed curve did not reach the target within the run.

Noise-subtracted columns (`*_beyond_noise`): the noise band of a metric is the min-max range of that metric over the
baseline runs (BENCHMARK.md, Stage 1). A run inside the band shows 0 (difference not distinguishable from noise).
A run outside shows its distance to the nearest band edge: positive = slower than the slowest baseline seed,
negative = faster than the fastest baseline seed. This is the difference "with the noise taken out".

Usage (from the repo root, with .venv active):
    python logs/learning_speed.py
    python logs/learning_speed.py --window 30 --csv logs/learning_speed.csv
"""

import argparse
import glob
import os
import sys

import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAG = "Train/mean_reward"
BASELINE_EXPERIMENT = "quadrustand_spot_flat_baseline"
LOGS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsl_rl")


def smooth(values, window):
    """Trailing moving average: point i is the mean of the last `window` points up to i (fewer at the start)."""
    csum = np.cumsum(np.insert(values, 0, 0.0))
    idx = np.arange(1, len(values) + 1)
    lo = np.maximum(idx - window, 0)
    return (csum[idx] - csum[lo]) / (idx - lo)


def first_reach(steps, smoothed, target):
    """First step where `smoothed` is >= target, or None if it never is."""
    hits = np.nonzero(smoothed >= target)[0]
    return int(steps[hits[0]]) if len(hits) else None


def beyond_noise(value, band):
    """Distance of `value` outside the noise band (lo, hi): 0 inside, +x above hi, -x below lo."""
    if value == "never":
        return "never"
    if band is None:
        return "n/a"
    lo, hi = band
    if value > hi:
        return f"+{value - hi}"
    if value < lo:
        return f"-{lo - value}"
    return 0


def load_run(run_dir):
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})  # 0 = keep every point
    acc.Reload()
    if TAG not in acc.Tags()["scalars"]:
        return None
    events = {e.step: e.value for e in acc.Scalars(TAG)}  # one value per step
    steps = np.array(sorted(events))
    values = np.array([events[s] for s in steps], dtype=float)
    seed = None
    agent_yaml = os.path.join(run_dir, "params", "agent.yaml")
    if os.path.exists(agent_yaml):
        with open(agent_yaml) as f:
            seed = yaml.safe_load(f).get("seed")
    return steps, values, seed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", type=int, default=20, help="moving-average window in iterations (default 20)")
    parser.add_argument("--csv", default=None, help="also write the table to this CSV file")
    args = parser.parse_args()

    run_dirs = sorted(d for d in glob.glob(os.path.join(LOGS_ROOT, "*", "*")) if os.path.isdir(d))
    if not run_dirs:
        sys.exit(f"No runs found under {LOGS_ROOT}")

    runs = []
    for d in run_dirs:
        loaded = load_run(d)
        if loaded is None:
            print(f"skipped (no {TAG}): {d}", file=sys.stderr)
            continue
        steps, values, seed = loaded
        n_tail = max(1, int(round(0.05 * len(values))))
        runs.append(
            {
                "variant": os.path.basename(os.path.dirname(d)).replace("quadrustand_spot_flat_", ""),
                "experiment": os.path.basename(os.path.dirname(d)),
                "run": os.path.basename(d),
                "seed": seed,
                "steps": steps,
                "smoothed": smooth(values, args.window),
                "final": float(values[-n_tail:].mean()),
                "last_step": int(steps[-1]),
            }
        )

    baseline_finals = [r["final"] for r in runs if r["experiment"] == BASELINE_EXPERIMENT]
    baseline_final = float(np.mean(baseline_finals)) if baseline_finals else None

    rows = []
    for r in runs:
        own = first_reach(r["steps"], r["smoothed"], 0.9 * r["final"]) if r["final"] > 0 else None
        base = (
            first_reach(r["steps"], r["smoothed"], 0.9 * baseline_final)
            if baseline_final is not None and baseline_final > 0
            else None
        )
        rows.append(
            {
                "variant": r["variant"],
                "seed": r["seed"],
                "run": r["run"],
                "last_step": r["last_step"],
                "final": round(r["final"], 3),
                "own_90": "never" if own is None else own,
                "baseline_90": "never" if base is None else base,
            }
        )

    # noise band per metric = min-max over the baseline runs (None if a baseline run never reached the target)
    baseline_label = BASELINE_EXPERIMENT.replace("quadrustand_spot_flat_", "")
    bands = {}
    for metric in ("own_90", "baseline_90"):
        vals = [row[metric] for row in rows if row["variant"] == baseline_label]
        bands[metric] = (min(vals), max(vals)) if vals and all(isinstance(v, int) for v in vals) else None
    for row in rows:
        for metric in ("own_90", "baseline_90"):
            row[f"{metric}_beyond_noise"] = beyond_noise(row[metric], bands[metric])

    rows.sort(key=lambda x: (x["variant"] != baseline_label, x["variant"], x["seed"] or 0))
    cols = [
        "variant", "seed", "run", "last_step", "final",
        "own_90", "own_90_beyond_noise", "baseline_90", "baseline_90_beyond_noise",
    ]  # fmt: skip
    widths = {c: max(len(c), *(len(str(row[c])) for row in rows)) for c in cols}
    print(f"smoothing window: {args.window} iterations (trailing mean); final = mean of last 5% of iterations")
    if baseline_final is not None:
        print(f"baseline reference final value: {baseline_final:.3f} (mean of {len(baseline_finals)} baseline run(s))")
    for metric, band in bands.items():
        text = "n/a" if band is None else f"{band[0]} .. {band[1]}"
        print(f"noise band {metric}: {text} (min..max over the baseline runs)")
    print()
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for row in rows:
        print("  ".join(str(row[c]).ljust(widths[c]) for c in cols))

    if args.csv:
        with open(args.csv, "w") as f:
            f.write(",".join(cols) + "\n")
            for row in rows:
                f.write(",".join(str(row[c]) for c in cols) + "\n")
        print(f"\nwritten: {args.csv}")


if __name__ == "__main__":
    main()
