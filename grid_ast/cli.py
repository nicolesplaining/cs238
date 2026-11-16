"""Run with `python -m grid_ast.cli --help` or the grid-ast entry point."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .model import Config
from .policies import load_policy


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a PPO battery policy")
    train.add_argument("--steps", type=int, default=32768)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--out", default="runs/training_seed7")
    train.add_argument("--config")
    for name in ("evaluate", "benchmark", "replay", "sensitivity"):
        p = commands.add_parser(name)
        p.add_argument("--config")
        p.add_argument("--model", required=True)
        p.add_argument("--out", default=f"runs/{name}")
        if name in ("evaluate", "sensitivity"):
            p.add_argument("--episodes", type=int, default=200)
            p.add_argument("--seed", type=int, default=20000)
            if name == "sensitivity":
                p.add_argument("--load-scales", default=".12,.30,.45")
        elif name == "benchmark":
            p.add_argument("--budget", type=int, default=6000)
            p.add_argument("--seeds", default="1000,1001,1002,1003,1004")
            p.add_argument("--exploration", type=float, default=.3)
            p.add_argument("--persistent-rollout-probability", type=float, default=.25)
        else:
            p.add_argument("--result", required=True)
            p.add_argument("--shield", action="store_true")
    plot = commands.add_parser("plot")
    plot.add_argument("--results", required=True)
    args = parser.parse_args()
    if args.command == "plot":
        from .plots import plot_results
        plot_results(args.results)
        return
    config = Config.load(args.config) if args.config else Config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "train":
        from .train import train
        print(json.dumps(train(config, out, args.steps, args.seed), indent=2))
        return
    policy = load_policy("ppo", args.model)
    if args.command == "evaluate":
        from .evaluate import evaluate, write_csv, paired_difference
        config.save(out / "config.json")
        results = {}
        for name in ("idle", "rule", "ppo", "ppo_shield"):
            controller = policy if name.startswith("ppo") else load_policy(name)
            result = evaluate(config, controller, args.episodes, args.seed, name == "ppo_shield")
            write_csv(out / f"{name}_episodes.csv", result["rows"])
            results[name] = result
            print(name, result["failure_rate"], result["mean_cost"], flush=True)
        delta = paired_difference(results["ppo"], results["ppo_shield"])
        summary = {k: {a:b for a,b in v.items() if a != "rows"} for k,v in results.items()}
        summary["paired_shield_effect"] = delta
        dump(out / "summary.json", summary)
    elif args.command == "sensitivity":
        from .evaluate import evaluate, write_csv, paired_difference
        if not args.config:
            config = replace(config, horizon=32, start_step=24)
        for sigma in [float(x) for x in args.load_scales.split(",")]:
            scenario = replace(config, load_sigma=sigma)
            folder = out / f"load_sigma_{sigma:.2f}"
            folder.mkdir(parents=True, exist_ok=True)
            scenario.save(folder / "config.json")
            runs = {}
            for shield in (False, True):
                result = evaluate(scenario, policy, args.episodes, args.seed, shield)
                name = "ppo_shield" if shield else "ppo"
                write_csv(folder / f"{name}_episodes.csv", result["rows"])
                runs[name] = result
                print(sigma, name, result["failed_episodes"], flush=True)
            summary = {k: {a:b for a,b in v.items() if a != "rows"} for k,v in runs.items()}
            summary["paired_shield_effect"] = paired_difference(runs["ppo"], runs["ppo_shield"])
            dump(folder / "summary.json", summary)
    elif args.command == "benchmark":
        from .search import search
        from .evaluate import write_csv
        if not args.config:
            config = replace(config, horizon=32, start_step=24)
        config.save(out / "config.json")
        rows = []
        for shield in (False, True):
            for method in ("mc", "mcts"):
                for seed in [int(x) for x in args.seeds.split(",")]:
                    result = search(config, policy, method, args.budget, seed, shield, args.exploration,
                                    args.persistent_rollout_probability)
                    name = f"{method}_{'shield' if shield else 'raw'}_seed{seed}"
                    dump(out / f"{name}.json", result)
                    best = result["best"]
                    row = {k:v for k,v in result.items() if k not in ("history", "best", "interpretation", "failure_modes")}
                    row.update({"best_log_probability": best["log_probability"] if best else None,
                                "best_failure_length": len(best["sequence"]) if best else None,
                                "best_found_at_transition": best["found_at_transition"] if best else None,
                                "first_failure_at_transition": next((r["transitions"] for r in result["history"] if r["failures"]), None),
                                "distinct_modes": len(result["failure_modes"])})
                    rows.append(row)
                    write_csv(out / "summary.csv", rows)
                    print(name, row["failed_rollouts"], row["best_log_probability"], flush=True)
    elif args.command == "replay":
        from .search import replay
        result = json.loads(Path(args.result).read_text())
        if result.get("best") is None:
            raise ValueError("The selected search run contains no failure to replay.")
        if not args.config:
            config = Config.load(Path(args.result).parent / "config.json")
        rows = replay(config, policy, result["best"]["sequence"], args.shield)
        dump(out / "replay.json", rows)
        print(json.dumps(rows[-1], indent=2))


if __name__ == "__main__":
    main()
