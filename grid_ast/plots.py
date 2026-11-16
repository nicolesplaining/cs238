"""Rebuild experiment figures from saved JSON/CSV, without rerunning training."""

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"mc": "#8793a6", "mcts": "#b33b40"}


def plot_results(results):
    root = Path(results)
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "font.family": "DejaVu Sans"})
    summary_path = root / "evaluation_morning" / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        names = ["idle", "rule", "ppo", "ppo_shield"]
        labels = ["Idle", "Rule", "PPO", "PPO + shield"]
        rates = np.array([summary[n]["failure_rate"] for n in names])
        bounds = np.array([summary[n]["failure_rate_wilson95"] for n in names])
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), layout="constrained")
        axes[0].bar(labels, 100*rates, color=["#8793a6", "#91b7a5", "#b33b40", "#e0a84f"])
        axes[0].errorbar(np.arange(4), 100*rates,
                        yerr=100*np.maximum(0, np.vstack([rates-bounds[:,0],bounds[:,1]-rates])),
                        fmt="none", color="#252a34", capsize=4)
        axes[0].set_ylabel("Episodes with a voltage failure (%)")
        axes[0].set_title("Nominal Monte Carlo, Wilson 95% intervals")
        costs = [summary[n]["mean_cost"] for n in names]
        axes[1].bar(labels, costs, color=["#8793a6", "#91b7a5", "#b33b40", "#e0a84f"])
        axes[1].errorbar(np.arange(4), costs, yerr=[1.96*summary[n]["cost_standard_error"] for n in names],
                        fmt="none", color="#252a34", capsize=4)
        axes[1].set_ylabel("Mean import + throughput cost ($)")
        axes[1].set_title("Synthetic tariff; 8-hour morning window")
        fig.savefig(out / "nominal_evaluation.png")
        plt.close(fig)
    search_folder = root / "search_persistent"
    if not search_folder.exists():
        search_folder = root / "search"
    files = sorted(search_folder.glob("*seed*.json"))
    if files:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), layout="constrained", sharey=True)
        runs = [json.loads(p.read_text()) for p in files]
        for shield, ax in zip((False, True), axes):
            for method in ("mc", "mcts"):
                subset = [r for r in runs if r["shielded"] == shield and r["method"] == method]
                plotted = False
                for i,r in enumerate(subset):
                    hist = r["history"]
                    x = [h["transitions"] for h in hist]
                    y = [h["best_log_probability"] if h["best_log_probability"] is not None else np.nan for h in hist]
                    if np.any(np.isfinite(y)):
                        ax.step(x, y, where="post", color=COLORS[method], alpha=.65, label=method.upper() if not plotted else None)
                        plotted = True
                if subset and not plotted:
                    ax.plot([], [], color=COLORS[method], label=f"{method.upper()}: no failures found")
            ax.set_title("PPO + forecast shield" if shield else "PPO")
            ax.set_xlabel("Actual environment transitions, including prefix replay")
            ax.legend(frameon=False)
        axes[0].set_ylabel("Best-found failure log probability\n(higher is more likely)")
        fig.savefig(out / "search_progress.png")
        plt.close(fig)
    monitor = root / "training_seed7" / "episodes.monitor.csv"
    if monitor.exists():
        with monitor.open() as f:
            next(f)
            rows = list(csv.DictReader(f))
        rewards = np.array([float(r["r"]) for r in rows])
        steps = np.cumsum([int(r["l"]) for r in rows])
        fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
        ax.plot(steps, rewards, alpha=.25, color="#8793a6", label="Episode return")
        if len(rewards) >= 20:
            ax.plot(steps[19:], np.convolve(rewards,np.ones(20)/20,mode="valid"), color="#b33b40", label="20-episode mean")
        ax.set(xlabel="Training transitions", ylabel="Undiscounted return", title="PPO training, seed 7")
        ax.legend(frameon=False)
        fig.savefig(out / "training.png")
        plt.close(fig)
    trace_path = root / "failure_replay.json"
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
        fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.0), layout="constrained", sharex=True)
        for name, color in (("raw", "#b33b40"), ("shield", "#315d8c")):
            rows = trace[name]
            x = [r["elapsed"]*.25 for r in rows]
            style = "--" if name == "shield" else "-"
            axes[0].plot(x, [min(r["voltages"]) for r in rows], marker="o", linestyle=style, markersize=3, color=color, label=name)
            axes[1].plot(x, [r["soc"] for r in rows], linestyle=style, color=color, label=name)
            axes[2].plot(x, [r["actual_charge_mw"] for r in rows], linestyle=style, color=color, label=name)
        axes[0].axhline(.95, linestyle="--", color="black", linewidth=1)
        axes[0].set(ylabel="Minimum voltage (p.u.)", title="Replay of one raw-policy failure prefix")
        axes[0].legend(frameon=False)
        axes[1].set_ylabel("Battery SoC")
        axes[2].set(ylabel="Charging power (MW)", xlabel="Hours after 06:00 initial state")
        fig.savefig(out / "failure_replay.png")
        plt.close(fig)
