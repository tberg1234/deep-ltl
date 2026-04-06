#!/usr/bin/env python
"""Plot training logs from experiments/ppo/<env>/<name>/<seed>/log.csv"""
import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_avg_goal_success(s: str) -> dict[str, float]:
    """Extract {object_name: success_rate} from the avg_goal_success string."""
    result = {}
    for m in re.finditer(r"frozenset\(\{(\w+)\}\)[^:]*:\s*([\d.]+)", s):
        result[m.group(1)] = float(m.group(2))
    return result


def load_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    goal_dicts = df["avg_goal_success"].apply(parse_avg_goal_success)
    goal_df = pd.DataFrame(list(goal_dicts))
    return pd.concat([df.drop(columns=["avg_goal_success"]), goal_df], axis=1)


def plot_run(ax_map: dict, df: pd.DataFrame, label: str):
    x = df["num_steps"]
    for key, ax in ax_map.items():
        if key in df.columns:
            ax.plot(x, df[key], label=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="SafetyBallNav-v0")
    parser.add_argument("--names", nargs="+", required=True,
                        help="Experiment names, e.g. my_run my_run2")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--smooth", type=int, default=10,
                        help="Rolling average window (0 = no smoothing)")
    args = parser.parse_args()

    base = Path("experiments/ppo") / args.env

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle(f"{args.env} — Training Logs")

    object_colors = {}
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    metrics = {
        "success_per_episode_mean": (axes[0, 0], "Success rate"),
        "violation_per_episode_mean": (axes[0, 1], "Violation rate"),
        "return_per_episode_mean": (axes[0, 2], "Return"),
        "entropy": (axes[1, 0], "Entropy"),
        "policy_loss": (axes[1, 1], "Policy loss"),
        "value_loss": (axes[1, 2], "Value loss"),
        "num_steps_per_episode_mean": (axes[2, 0], "Steps / episode"),
        "grad_norm": (axes[2, 1], "Gradient norm"),
        "adr": (axes[2, 2], "Avg discounted return"),
    }
    ax_map = {k: v[0] for k, v in metrics.items()}
    per_task_ax = None  # will be set below

    # Per-task success — replace one subplot
    per_task_ax = axes[2, 2]
    del ax_map["adr"]

    for ax, title in [(v[0], v[1]) for v in metrics.values()]:
        ax.set_title(title)
        ax.set_xlabel("Steps")

    per_task_ax.set_title("Per-task success rate")
    per_task_ax.set_xlabel("Steps")

    object_names = set()
    run_dfs = []

    for name in args.names:
        for seed in args.seeds:
            log_path = base / name / str(seed) / "log.csv"
            if not log_path.exists():
                print(f"Not found: {log_path}")
                continue
            df = load_log(log_path)
            if args.smooth > 1:
                numeric = df.select_dtypes(include="number")
                df[numeric.columns] = numeric.rolling(args.smooth, min_periods=1).mean()
            label = f"{name}/{seed}"
            run_dfs.append((label, df))
            # collect object column names
            task_cols = [c for c in df.columns if c not in metrics and
                         c not in ("num_steps", "remaining", "sps", "arps",
                                   "return_per_episode_std", "success_per_episode_std",
                                   "violation_per_episode_std", "num_steps_per_episode_std",
                                   "value")]
            object_names.update(task_cols)

    # Assign colors to objects
    for i, obj in enumerate(sorted(object_names)):
        object_colors[obj] = color_cycle[i % len(color_cycle)]

    for label, df in run_dfs:
        x = df["num_steps"]
        for key, ax in ax_map.items():
            if key in df.columns:
                ax.plot(x, df[key], label=label)
        # per-task
        for obj in sorted(object_names):
            if obj in df.columns:
                per_task_ax.plot(x, df[obj], color=object_colors[obj],
                                 label=obj if label == run_dfs[0][0] else "_nolegend_",
                                 linestyle="-" if label == run_dfs[0][0] else "--")

    for ax, _ in metrics.values():
        if ax.lines:
            ax.legend(fontsize=7)
    per_task_ax.legend(fontsize=7)

    plt.tight_layout()
    out_dir = Path("plots") / args.env
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{'_'.join(args.names)}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
