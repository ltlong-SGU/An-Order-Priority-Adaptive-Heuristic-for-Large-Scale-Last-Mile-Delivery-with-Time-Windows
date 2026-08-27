# -*- coding: utf-8 -*-
"""E5: fair external/control comparison for Homberger-1000 VRPTW.

Reads E1, E2, E3, E4, and the original benchmark ZIP directly.  Compares:

  * BIH baseline from E2;
  * conventional adaptive large-neighbourhood search (ALNS), rerun under the
    same starting solution, seeds, iteration cap, and time budget as DA-ADR;
  * proposed DA-ADR results already obtained in E4; and
  * published SINTEF best-known solutions (BKS), used only as reference values.

No PyVRP result is fabricated and no additional solver installation is needed.
The standard ALNS control has no access to the E3 P4 priority score.  All
reported routes are checked independently using double-precision Euclidean
distances, customer coverage, vehicle capacity, and time-window constraints.

PowerShell:
    python E5_External_Benchmark_Comparison.py --quick
    python E5_External_Benchmark_Comparison.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import logging
import math
import platform
import random
import subprocess
import sys
import time
import types
import zipfile
from datetime import datetime
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as exc:
    print(f"Missing required package: {exc}")
    print("Install using: python -m pip install numpy pandas matplotlib openpyxl")
    raise SystemExit(1) from exc

try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except ImportError:
    wilcoxon = None
    SCIPY_AVAILABLE = False


ROOT = (
    r"F:\dulieu\Nghien cứu sinh\Nghiên cứu sinh chính thức"
    r"\Paper_An Order-Priority Adaptive Heuristic for Large-Scale Last-Mile Delivery with Time Windows"
    r"\Empirical"
)
DEFAULT_E1 = Path(ROOT + r"\E1.zip")
DEFAULT_E2 = Path(ROOT + r"\E2.zip")
DEFAULT_E3 = Path(ROOT + r"\E3.zip")
DEFAULT_E4 = Path(ROOT + r"\E4.zip")
DEFAULT_DATASET = Path(ROOT + r"\homberger_1000_customer_instances.zip")
DEFAULT_OUTPUT = Path(ROOT + r"\E5")
FAMILIES = ("C1", "C2", "R1", "R2", "RC1", "RC2")
CONTROL_OPERATORS = ("random", "related", "worst_arc", "route_elimination")
BKS_SOURCE = "https://www.sintef.no/projectweb/top/vrptw/1000-customers/"
BKS_ACCESSED = "2026-08-25"
EPS = 1e-8

# Snapshot transcribed from the official SINTEF Gehring--Homberger 1000-customer
# hierarchical benchmark.  Columns: family -> instance 1..10 -> (vehicles, distance).
# SINTEF requires double-precision distances and compares vehicles before distance.
SINTEF_BKS = {
    "C1": [(100, 42478.95), (90, 42222.96), (90, 40101.36), (90, 39468.60),
           (100, 42469.18), (99, 43830.21), (97, 43341.77), (92, 42629.91),
           (90, 40318.03), (90, 39852.44)],
    "C2": [(30, 16879.24), (29, 17126.39), (28, 16829.47), (28, 15607.48),
           (30, 16561.29), (29, 16863.71), (29, 17602.84), (28, 16512.43),
           (28, 17809.34), (28, 15937.45)],
    "R1": [(100, 53380.18), (91, 48232.67), (91, 44694.16), (91, 42463.74),
           (91, 50445.39), (91, 46929.17), (91, 43975.47), (91, 42288.50),
           (91, 49195.26), (91, 47407.16)],
    "R2": [(19, 42182.57), (19, 33411.21), (19, 24916.88), (19, 17851.96),
           (19, 36216.05), (19, 29978.02), (19, 23219.61), (19, 17442.29),
           (19, 32995.71), (19, 30207.49)],
    "RC1": [(90, 45830.62), (90, 43718.84), (90, 42146.79), (90, 41391.18),
            (90, 45069.37), (90, 44937.36), (90, 44457.79), (90, 43956.91),
            (90, 43897.21), (90, 43551.69)],
    "RC2": [(20, 30276.27), (18, 26104.09), (18, 19911.48), (18, 15693.28),
            (18, 27067.04), (18, 26741.27), (18, 24999.66), (18, 23595.33),
            (18, 22919.42), (18, 21834.94)],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_member(zf: zipfile.ZipFile, suffix: str) -> str:
    normalized = suffix.replace("\\", "/").lower()
    matches = [name for name in zf.namelist() if name.replace("\\", "/").lower().endswith(normalized)]
    if not matches:
        raise FileNotFoundError(f"Missing {suffix!r} in {zf.filename}")
    return min(matches, key=len)


def read_json(zf: zipfile.ZipFile, suffix: str) -> dict:
    return json.loads(zf.read(zip_member(zf, suffix)).decode("utf-8-sig"))


def read_csv(zf: zipfile.ZipFile, suffix: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(zf.read(zip_member(zf, suffix))))


def setup_folders(root: Path) -> dict[str, Path]:
    folders = {
        "root": root,
        "tables": root / "tables",
        "charts": root / "charts",
        "maps": root / "route_maps",
        "solutions": root / "standard_alns_best_solutions_json",
        "checkpoints": root / "checkpoints",
    }
    for path in folders.values():
        path.mkdir(parents=True, exist_ok=True)
    return folders


def setup_logger(root: Path) -> logging.Logger:
    logger = logging.getLogger("E5")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(root / "E5_execution.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger


def load_e4_engine(e4_path: Path):
    """Load the already audited E4 feasibility/neighbourhood engine in memory."""
    with zipfile.ZipFile(e4_path) as zf:
        member = zip_member(zf, "E4_DA_ADR_Optimizer.py")
        source = zf.read(member).decode("utf-8-sig")
    name = "e5_audited_e4_engine"
    module = types.ModuleType(name)
    module.__file__ = str(e4_path) + ":" + member
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def verify_lineage(args: argparse.Namespace) -> tuple[dict, dict]:
    for path in (args.e1, args.e2, args.e3, args.e4, args.dataset):
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
    manifests = {}
    for stage, path in [("E1", args.e1), ("E2", args.e2), ("E3", args.e3), ("E4", args.e4)]:
        with zipfile.ZipFile(path) as zf:
            manifests[stage] = read_json(zf, f"{stage}_reproducibility_manifest.json")
    actual_hash = sha256_file(args.dataset)
    recorded = {stage: manifest.get("dataset_sha256") for stage, manifest in manifests.items()}
    if any(value != actual_hash for value in recorded.values()):
        raise ValueError(f"Dataset SHA-256 differs across stages: actual={actual_hash}; recorded={recorded}")
    if not manifests["E4"].get("all_feasible", False):
        raise ValueError("E4 manifest does not certify feasible solutions")
    if not manifests["E4"].get("all_non_degraded", False):
        raise ValueError("E4 manifest does not certify the non-degradation safeguard")
    return {"dataset_sha256": actual_hash, "stage_dataset_sha256": recorded}, manifests


def bks_frame() -> pd.DataFrame:
    rows = []
    for family, values in SINTEF_BKS.items():
        for index, (vehicles, distance) in enumerate(values, start=1):
            rows.append({
                "instance": f"{family}_10_{index}",
                "family": family,
                "bks_vehicles": vehicles,
                "bks_distance": distance,
                "source": BKS_SOURCE,
                "access_date": BKS_ACCESSED,
                "objective": "lexicographic: vehicles, then double-precision distance",
            })
    return pd.DataFrame(rows)


def independent_validate(instance, routes: list[list[int]]) -> dict:
    """Independent validator intentionally does not call the E4 route evaluator."""
    served = []
    total_distance = 0.0
    capacity_violations = 0
    time_violations = 0
    depot_violations = 0
    for route in routes:
        if not route or route[0] != 0 or route[-1] != 0:
            depot_violations += 1
            continue
        clock = float(instance.ready[0])
        load = 0.0
        for previous, node in zip(route, route[1:]):
            travel = float(np.hypot(instance.x[previous] - instance.x[node], instance.y[previous] - instance.y[node]))
            total_distance += travel
            clock = max(float(instance.ready[node]), clock + float(instance.service[previous]) + travel)
            if clock > float(instance.due[node]) + EPS:
                time_violations += 1
            if node:
                load += float(instance.demand[node])
                served.append(node)
            if load > float(instance.capacity) + EPS:
                capacity_violations += 1
    expected = set(range(1, len(instance.x)))
    actual = set(served)
    missing = len(expected - actual)
    duplicates = len(served) - len(actual)
    feasible = not (capacity_violations or time_violations or depot_violations or missing or duplicates)
    return {
        "feasible": feasible,
        "vehicles": len([route for route in routes if len(route) > 2]),
        "distance": total_distance,
        "customers_served": len(served),
        "missing_customers": missing,
        "duplicate_customers": duplicates,
        "capacity_violations": capacity_violations,
        "time_window_violations": time_violations,
        "depot_violations": depot_violations,
    }


def standard_alns(engine, instance, baseline, seed: int, iterations: int, time_limit: float):
    """Conventional ALNS: generic operators + regret-2, no P4 information."""
    control = copy.copy(instance)
    control.difficulty = np.full_like(instance.difficulty, 0.5, dtype=float)
    control.difficulty[0] = 0.0
    rng = random.Random(seed)
    current = copy.deepcopy(baseline)
    best = copy.deepcopy(baseline)
    weights = {name: 1.0 for name in CONTROL_OPERATORS}
    uses = {name: 0 for name in CONTROL_OPERATORS}
    rewards = {name: 0.0 for name in CONTROL_OPERATORS}
    history = []
    started = time.perf_counter()
    temperature = max(1.0, 0.0025 * baseline.distance)
    stagnant = 0
    for iteration in range(1, iterations + 1):
        if time.perf_counter() - started >= time_limit:
            break
        operator = rng.choices(list(weights), weights=list(weights.values()), k=1)[0]
        fraction = min(0.15, rng.uniform(0.02, 0.06) * (1.35 if stagnant > 35 else 1.0))
        removed = engine.select_destroy(control, current, operator, fraction, rng)
        partial = engine.remove_customers(current.routes, removed)
        rebuilt = engine.repair(control, partial, removed, rng, "regret2")
        uses[operator] += 1
        reward = 0.0
        accepted = False
        candidate = None
        if rebuilt is not None:
            try:
                candidate = engine.evaluate(control, rebuilt)
            except ValueError:
                candidate = None
        if candidate is not None:
            if engine.better(candidate, best):
                best = copy.deepcopy(candidate)
                current = candidate
                reward, accepted, stagnant = 8.0, True, 0
            elif engine.better(candidate, current):
                current = candidate
                reward, accepted = 4.0, True
                stagnant += 1
            elif candidate.vehicles == current.vehicles:
                probability = math.exp(-max(0.0, candidate.distance - current.distance) / max(temperature, 1e-9))
                if rng.random() < probability:
                    current = candidate
                    reward, accepted = 1.0, True
                stagnant += 1
            else:
                stagnant += 1
        else:
            stagnant += 1
        rewards[operator] += reward
        weights[operator] = max(0.10, 0.80 * weights[operator] + 0.20 * (1.0 + reward))
        temperature *= 0.992
        history.append({
            "iteration": iteration,
            "elapsed_seconds": time.perf_counter() - started,
            "operator": operator,
            "repair": "regret2",
            "removed": len(removed),
            "accepted": accepted,
            "current_vehicles": current.vehicles,
            "current_distance": current.distance,
            "best_vehicles": best.vehicles,
            "best_distance": best.distance,
        })
    if not history:
        history.append({"iteration": 0, "elapsed_seconds": 0.0, "operator": "none", "repair": "none", "removed": 0,
                        "accepted": False, "current_vehicles": baseline.vehicles, "current_distance": baseline.distance,
                        "best_vehicles": baseline.vehicles, "best_distance": baseline.distance})
    operator_rows = [{"operator": name, "uses": uses[name], "total_reward": rewards[name], "final_weight": weights[name]}
                     for name in CONTROL_OPERATORS]
    return best, history, operator_rows


def compare_lexicographic(first_vehicles: int, first_distance: float, second_vehicles: int, second_distance: float) -> str:
    if first_vehicles < second_vehicles:
        return "win"
    if first_vehicles > second_vehicles:
        return "loss"
    if first_distance < second_distance - EPS:
        return "win"
    if first_distance > second_distance + EPS:
        return "loss"
    return "tie"


def sign_test_pvalue(difference: np.ndarray) -> float:
    nonzero = difference[np.abs(difference) > EPS]
    if not len(nonzero):
        return 1.0
    positives = int(np.sum(nonzero > 0))
    negatives = len(nonzero) - positives
    tail = sum(math.comb(len(nonzero), i) for i in range(min(positives, negatives) + 1)) / (2 ** len(nonzero))
    return min(1.0, 2.0 * tail)


def paired_test(first: np.ndarray, second: np.ndarray) -> tuple[str, float, float]:
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    nonzero = difference[np.abs(difference) > EPS]
    if not len(nonzero):
        return ("Wilcoxon signed-rank" if SCIPY_AVAILABLE else "Exact sign test", 0.0, 1.0)
    if SCIPY_AVAILABLE:
        result = wilcoxon(first, second, alternative="two-sided", zero_method="wilcox")
        return "Wilcoxon signed-rank", float(result.statistic), float(result.pvalue)
    return "Exact sign test", float(np.sum(nonzero > 0)), sign_test_pvalue(difference)


def holm_adjust(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    order = np.argsort(pvalues)
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def build_statistics(paired: pd.DataFrame) -> pd.DataFrame:
    # Five seeds of the same instance are repeated measurements, not five
    # statistically independent benchmark instances. Aggregate paired runs
    # first so the global test has n=60 (and each family n=10), avoiding
    # pseudoreplication and spuriously small p-values.
    independent = paired.groupby(["instance", "family"], as_index=False).agg(
        alns_vehicles=("alns_vehicles", "mean"),
        alns_distance=("alns_distance", "mean"),
        da_adr_vehicles=("da_adr_vehicles", "mean"),
        da_adr_distance=("da_adr_distance", "mean"),
        matched_seeds=("seed", "count"),
    )
    rows = []
    for family in ["ALL", *FAMILIES]:
        frame = independent if family == "ALL" else independent[independent.family == family]
        if frame.empty:
            continue
        for metric, a, b in [
            ("vehicles", "da_adr_vehicles", "alns_vehicles"),
            ("distance", "da_adr_distance", "alns_distance"),
        ]:
            name, statistic, pvalue = paired_test(frame[a].to_numpy(), frame[b].to_numpy())
            rows.append({
                "family": family,
                "metric": metric,
                "test": name,
                "paired_observations": len(frame),
                "matched_runs_aggregated": int(frame.matched_seeds.sum()),
                "statistic": statistic,
                "p_value": pvalue,
                "mean_da_adr": float(frame[a].mean()),
                "mean_standard_alns": float(frame[b].mean()),
                "median_difference_da_minus_alns": float((frame[a] - frame[b]).median()),
                "da_adr_wins": int(np.sum(frame[a] < frame[b] - EPS)),
                "ties": int(np.sum(np.abs(frame[a] - frame[b]) <= EPS)),
                "da_adr_losses": int(np.sum(frame[a] > frame[b] + EPS)),
            })
    stats = pd.DataFrame(rows)
    if not stats.empty:
        stats["holm_adjusted_p_value"] = holm_adjust(stats["p_value"].tolist())
        stats["significant_at_0_05"] = stats["holm_adjusted_p_value"] < 0.05
    return stats


def save_plot(fig, folder: Path, name: str) -> None:
    fig.savefig(folder / f"{name}.png", dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_charts(comparison: pd.DataFrame, paired: pd.DataFrame, convergence: pd.DataFrame,
                  e4_convergence: pd.DataFrame, charts: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    groups = comparison.groupby("family", sort=False).mean(numeric_only=True).reindex(FAMILIES)
    x = np.arange(len(FAMILIES))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for offset, key, label, color in [(-width, "bih_vehicles", "BIH", "#93a4b4"),
                                       (0, "alns_vehicles", "Standard ALNS", "#e59739"),
                                       (width, "da_adr_vehicles", "DA-ADR", "#2c73a8")]:
        ax.bar(x + offset, groups[key], width, label=label, color=color)
    ax.plot(x, groups["bks_vehicles"], "kD", markersize=5, label="SINTEF BKS")
    ax.set_xticks(x, FAMILIES); ax.set_ylabel("Mean number of vehicles"); ax.legend(ncol=2)
    ax.set_title("Fleet-size comparison under the hierarchical objective")
    save_plot(fig, charts, "01_vehicle_comparison_by_family")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for offset, key, label, color in [(-width, "bih_distance", "BIH", "#93a4b4"),
                                       (0, "alns_distance", "Standard ALNS", "#e59739"),
                                       (width, "da_adr_distance", "DA-ADR", "#2c73a8")]:
        ax.bar(x + offset, groups[key], width, label=label, color=color)
    ax.plot(x, groups["bks_distance"], "kD", markersize=5, label="SINTEF BKS")
    ax.set_xticks(x, FAMILIES); ax.set_ylabel("Mean distance"); ax.legend(ncol=2)
    ax.set_title("Distance comparison; fleet sizes must be interpreted separately")
    save_plot(fig, charts, "02_distance_comparison_by_family")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - 0.18, groups["alns_vehicle_gap_vs_bks"], 0.36, label="Standard ALNS", color="#e59739")
    ax.bar(x + 0.18, groups["da_adr_vehicle_gap_vs_bks"], 0.36, label="DA-ADR", color="#2c73a8")
    ax.set_xticks(x, FAMILIES); ax.set_ylabel("Additional vehicles above SINTEF BKS")
    ax.set_title("Fleet gap to the official best-known reference"); ax.legend()
    save_plot(fig, charts, "03_vehicle_gap_to_sintef_bks")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    data = []
    labels = []
    for family in FAMILIES:
        values = paired.loc[paired.family == family, "da_adr_gain_vs_alns_percent"].to_numpy()
        if len(values):
            data.append(values); labels.append(family)
    if data:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.axhline(0, color="#222", linewidth=0.8)
    ax.set_ylabel("DA-ADR distance gain over paired ALNS (%)")
    ax.set_title("Paired-seed comparison: proposed method versus conventional ALNS")
    save_plot(fig, charts, "04_paired_seed_gain_boxplot")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for frame, label, color in [(convergence, "Standard ALNS", "#e59739"),
                                (e4_convergence, "DA-ADR (E4)", "#2c73a8")]:
        samples = []
        if frame.empty:
            continue
        for _, part in frame.groupby(["instance", "seed"]):
            baseline = float(part.iloc[0]["baseline_distance"])
            progress = np.linspace(0.0, 1.0, len(part))
            gain = 100.0 * (baseline - part["best_distance"].to_numpy(float)) / baseline
            samples.append(np.interp(np.linspace(0.0, 1.0, 101), progress, gain))
        if samples:
            ax.plot(np.linspace(0.0, 1.0, 101), np.mean(samples, axis=0), label=label, color=color, linewidth=2)
    ax.set_xlabel("Normalized search progress"); ax.set_ylabel("Mean distance improvement over BIH (%)")
    ax.set_title("Convergence comparison with matched initialization and run budget"); ax.legend()
    save_plot(fig, charts, "05_alns_vs_da_adr_convergence")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    results = comparison.groupby(["family", "da_adr_vs_alns_lexicographic"]).size().unstack(fill_value=0).reindex(FAMILIES, fill_value=0)
    bottom = np.zeros(len(FAMILIES))
    for outcome, color in [("win", "#2c73a8"), ("tie", "#a6adb4"), ("loss", "#d86b5c")]:
        vals = results[outcome].to_numpy() if outcome in results else np.zeros(len(FAMILIES))
        ax.bar(x, vals, bottom=bottom, color=color, label=outcome.title()); bottom += vals
    ax.set_xticks(x, FAMILIES); ax.set_ylabel("Instances")
    ax.set_title("DA-ADR versus standard ALNS: lexicographic win/tie/loss"); ax.legend(ncol=3)
    save_plot(fig, charts, "06_lexicographic_win_tie_loss")


def route_map(instance, baseline, alns, proposed, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.9), sharex=True, sharey=True)
    for ax, solution, title in zip(axes, [baseline, alns, proposed], ["E2 BIH", "E5 standard ALNS", "E4 DA-ADR"]):
        cmap = plt.get_cmap("tab20")
        for index, route in enumerate(solution.routes):
            nodes = np.asarray(route, dtype=int)
            ax.plot(instance.x[nodes], instance.y[nodes], color=cmap(index % 20), linewidth=0.55, alpha=0.75)
        ax.scatter(instance.x[1:], instance.y[1:], s=3.5, color="#34495e", alpha=0.5)
        ax.scatter(instance.x[0], instance.y[0], s=145, marker="*", color="#00a878", edgecolor="black", zorder=5)
        ax.set_title(f"{title}\nvehicles={solution.vehicles}, distance={solution.distance:.1f}")
        ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("X"); ax.grid(alpha=0.12)
    axes[0].set_ylabel("Y")
    fig.suptitle(f"{instance.name}: delivery-route comparison")
    save_plot(fig, path.parent, path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E5 external/control comparison for Homberger1000")
    parser.add_argument("--e1", type=Path, default=DEFAULT_E1)
    parser.add_argument("--e2", type=Path, default=DEFAULT_E2)
    parser.add_argument("--e3", type=Path, default=DEFAULT_E3)
    parser.add_argument("--e4", type=Path, default=DEFAULT_E4)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=int, default=0, help="0 matches the E4 number of runs")
    parser.add_argument("--iterations", type=int, default=0, help="0 matches the E4 iteration cap")
    parser.add_argument("--time-limit", type=float, default=0.0, help="0 matches the E4 per-run budget")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--limit-instances", type=int, default=0)
    parser.add_argument("--restart", action="store_true", help="Ignore saved runs and start again")
    parser.add_argument("--quick", action="store_true", help="One instance, one seed, 10 iterations, 12 seconds")
    return parser.parse_args()


def save_alns_solution(path: Path, instance, solution, baseline, seed: int, audit: dict) -> None:
    payload = {
        "instance": instance.name,
        "family": instance.family,
        "algorithm": "Standard ALNS",
        "seed": seed,
        "objective": {"vehicle_count": solution.vehicles, "total_distance": solution.distance},
        "e2_baseline": {"vehicle_count": baseline.vehicles, "total_distance": baseline.distance},
        "uses_e3_p4": False,
        "validation": audit,
        "routes": solution.routes,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_checkpoint_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_run_checkpoint(path: Path, signature: dict, instance_name: str, seed: int):
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("signature") != signature:
        return None
    if payload.get("instance") != instance_name or payload.get("seed") != seed:
        return None
    required = ("routes", "history", "operator_stats", "runtime_seconds")
    return payload if all(key in payload for key in required) else None


def main() -> int:
    args = parse_args()
    if args.quick:
        args.limit_instances, args.runs, args.iterations, args.time_limit = 1, 1, 10, 12.0
        args.output = args.output.parent / (args.output.name + "_quick_test")
    folders = setup_folders(args.output)
    logger = setup_logger(args.output)
    started = time.perf_counter()
    logger.info("Starting E5: BIH vs standard ALNS vs DA-ADR vs official SINTEF BKS")
    for name in ("e1", "e2", "e3", "e4", "dataset", "output"):
        logger.info("%s = %s", name.upper(), getattr(args, name))
    try:
        lineage, manifests = verify_lineage(args)
        e4_manifest = manifests["E4"]
        args.runs = args.runs or int(e4_manifest.get("runs_per_instance", 5))
        args.iterations = args.iterations or int(e4_manifest.get("iterations", 300))
        args.time_limit = args.time_limit or float(e4_manifest.get("time_limit_seconds_per_run", 45.0))
        base_seed = int(e4_manifest.get("base_seed", 2026))
        engine = load_e4_engine(args.e4)
        instances, e2_solutions = engine.load_data(args.e1, args.e2, args.e3, set(args.families), args.limit_instances)
        with zipfile.ZipFile(args.e4) as z4:
            e4_runs = read_csv(z4, "E4_all_runs.csv")
            e4_convergence = read_csv(z4, "E4_convergence_history.csv")
            e4_solutions = {}
            for name in z4.namelist():
                if name.lower().endswith("_da_adr_best.json"):
                    data = json.loads(z4.read(name).decode("utf-8-sig"))
                    e4_solutions[data["instance"]] = data
        names = {instance.name for instance in instances}
        e4_runs = e4_runs[e4_runs.instance.isin(names)].copy()
        e4_convergence = e4_convergence[e4_convergence.instance.isin(names)].copy()
        bks = bks_frame()
        bks_lookup = bks.set_index("instance")
        logger.info("Validated matching dataset SHA-256: %s", lineage["dataset_sha256"])
        logger.info("Matched protocol: instances=%d, runs=%d, iterations=%d, time-limit=%.1fs/run",
                    len(instances), args.runs, args.iterations, args.time_limit)
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not installed: exact paired sign test will replace Wilcoxon")

        alns_run_rows, paired_rows, comparison_rows, convergence_rows, operator_rows, validation_rows = [], [], [], [], [], []
        representatives = {}
        run_checkpoint_folder = folders["checkpoints"] / "completed_runs"
        run_checkpoint_folder.mkdir(parents=True, exist_ok=True)
        signature = {
            "version": 1, "dataset_sha256": lineage["dataset_sha256"],
            "runs": args.runs, "iterations": args.iterations,
            "time_limit": args.time_limit, "base_seed": base_seed,
        }
        if args.restart:
            for old_checkpoint in run_checkpoint_folder.glob("*.json"):
                old_checkpoint.unlink()
            logger.info("Restart requested: previous saved runs were removed")
        else:
            logger.info("Automatic resume enabled: completed runs will be reused")
        for index, instance in enumerate(instances, start=1):
            if instance.name not in e4_solutions:
                raise ValueError(f"Missing E4 solution for {instance.name}")
            e2_routes = [[int(n) for n in route] for route in e2_solutions[instance.name]["routes"]]
            baseline = engine.evaluate(instance, e2_routes)
            proposed_routes = [[int(n) for n in route] for route in e4_solutions[instance.name]["routes"]]
            proposed = engine.evaluate(instance, proposed_routes)
            for algorithm, solution in [("BIH", baseline), ("DA-ADR", proposed)]:
                audit = independent_validate(instance, solution.routes)
                if not audit["feasible"] or abs(audit["distance"] - solution.distance) > 1e-5:
                    raise ValueError(f"Independent {algorithm} validation failed for {instance.name}: {audit}")
                validation_rows.append({"instance": instance.name, "family": instance.family, "algorithm": algorithm, **audit})

            best_alns = copy.deepcopy(baseline)
            best_seed = base_seed
            for run in range(args.runs):
                seed = base_seed + run
                run_checkpoint = run_checkpoint_folder / f"{instance.name}_seed_{seed}.json"
                saved_run = load_run_checkpoint(run_checkpoint, signature, instance.name, seed)
                if saved_run is not None:
                    saved_routes = [[int(node) for node in route] for route in saved_run["routes"]]
                    candidate = engine.evaluate(instance, saved_routes)
                    history = saved_run["history"]
                    operator_stats = saved_run["operator_stats"]
                    elapsed = float(saved_run["runtime_seconds"])
                    logger.info("RESUME [%02d/%02d] %s run=%d/%d seed=%d: reused saved result",
                                index, len(instances), instance.name, run + 1, args.runs, seed)
                else:
                    run_start = time.perf_counter()
                    candidate, history, operator_stats = standard_alns(engine, instance, baseline, seed,
                                                                        args.iterations, args.time_limit)
                    elapsed = time.perf_counter() - run_start
                audit = independent_validate(instance, candidate.routes)
                if not audit["feasible"] or abs(audit["distance"] - candidate.distance) > 1e-5:
                    raise ValueError(f"Independent standard ALNS validation failed for {instance.name}: {audit}")
                if saved_run is None:
                    write_checkpoint_atomic(run_checkpoint, {
                        "signature": signature, "instance": instance.name,
                        "family": instance.family, "seed": seed, "run": run + 1,
                        "routes": [[int(node) for node in route] for route in candidate.routes],
                        "history": history, "operator_stats": operator_stats,
                        "runtime_seconds": elapsed,
                        "saved_at": datetime.now().isoformat(timespec="seconds"),
                    })
                if engine.better(candidate, best_alns):
                    best_alns, best_seed = copy.deepcopy(candidate), seed
                da_match = e4_runs[(e4_runs.instance == instance.name) & (e4_runs.seed == seed)]
                if da_match.empty:
                    raise ValueError(f"E4 has no matched seed {seed} for {instance.name}; reduce --runs")
                da = da_match.iloc[0]
                alns_run_rows.append({
                    "instance": instance.name, "family": instance.family, "run": run + 1, "seed": seed,
                    "baseline_vehicles": baseline.vehicles, "baseline_distance": baseline.distance,
                    "alns_vehicles": candidate.vehicles, "alns_distance": candidate.distance,
                    "vehicle_reduction_vs_bih": baseline.vehicles - candidate.vehicles,
                    "distance_gain_vs_bih_percent": 100.0 * (baseline.distance - candidate.distance) / baseline.distance,
                    "iterations_completed": history[-1]["iteration"], "runtime_seconds": elapsed,
                    "feasible": audit["feasible"], "uses_e3_p4": False,
                })
                paired_rows.append({
                    "instance": instance.name, "family": instance.family, "run": run + 1, "seed": seed,
                    "baseline_vehicles": baseline.vehicles, "baseline_distance": baseline.distance,
                    "alns_vehicles": candidate.vehicles, "alns_distance": candidate.distance,
                    "alns_runtime_seconds": elapsed,
                    "da_adr_vehicles": int(da.best_vehicles), "da_adr_distance": float(da.best_distance),
                    "da_adr_runtime_seconds": float(da.runtime_seconds),
                    "da_adr_gain_vs_alns_percent": 100.0 * (candidate.distance - float(da.best_distance)) / candidate.distance,
                    "da_adr_vs_alns_lexicographic": compare_lexicographic(int(da.best_vehicles), float(da.best_distance),
                                                                              candidate.vehicles, candidate.distance),
                })
                for row in history:
                    convergence_rows.append({"instance": instance.name, "family": instance.family, "run": run + 1,
                                             "seed": seed, "baseline_distance": baseline.distance, **row})
                for row in operator_stats:
                    operator_rows.append({"instance": instance.name, "family": instance.family, "run": run + 1,
                                          "seed": seed, **row})
                logger.info("[%02d/%02d] %-11s run=%d/%d | BIH %d/%.1f | ALNS %d/%.1f | DA-ADR %d/%.1f | %.1fs",
                            index, len(instances), instance.name, run + 1, args.runs,
                            baseline.vehicles, baseline.distance, candidate.vehicles, candidate.distance,
                            int(da.best_vehicles), float(da.best_distance), elapsed)
                write_checkpoint_atomic(folders["checkpoints"] / "E5_progress.json", {
                    "current_instance": instance.name, "current_index": index,
                    "completed_run": run + 1, "runs_per_instance": args.runs,
                    "total_instances": len(instances),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                })

            audit = independent_validate(instance, best_alns.routes)
            validation_rows.append({"instance": instance.name, "family": instance.family, "algorithm": "Standard ALNS", **audit})
            save_alns_solution(folders["solutions"] / f"{instance.name}_standard_ALNS_best.json",
                               instance, best_alns, baseline, best_seed, audit)
            ref = bks_lookup.loc[instance.name]
            bks_vehicles, bks_distance = int(ref.bks_vehicles), float(ref.bks_distance)
            row = {
                "instance": instance.name, "family": instance.family,
                "bks_vehicles": bks_vehicles, "bks_distance": bks_distance,
                "bih_vehicles": baseline.vehicles, "bih_distance": baseline.distance,
                "alns_vehicles": best_alns.vehicles, "alns_distance": best_alns.distance,
                "da_adr_vehicles": proposed.vehicles, "da_adr_distance": proposed.distance,
                "bih_vehicle_gap_vs_bks": baseline.vehicles - bks_vehicles,
                "alns_vehicle_gap_vs_bks": best_alns.vehicles - bks_vehicles,
                "da_adr_vehicle_gap_vs_bks": proposed.vehicles - bks_vehicles,
                "bih_distance_gap_vs_bks_percent": 100.0 * (baseline.distance - bks_distance) / bks_distance,
                "alns_distance_gap_vs_bks_percent": 100.0 * (best_alns.distance - bks_distance) / bks_distance,
                "da_adr_distance_gap_vs_bks_percent": 100.0 * (proposed.distance - bks_distance) / bks_distance,
                "alns_equal_fleet_distance_gap_percent": (
                    100.0 * (best_alns.distance - bks_distance) / bks_distance if best_alns.vehicles == bks_vehicles else np.nan
                ),
                "da_adr_equal_fleet_distance_gap_percent": (
                    100.0 * (proposed.distance - bks_distance) / bks_distance if proposed.vehicles == bks_vehicles else np.nan
                ),
                "da_adr_gain_vs_alns_percent": 100.0 * (best_alns.distance - proposed.distance) / best_alns.distance,
                "da_adr_vs_alns_lexicographic": compare_lexicographic(proposed.vehicles, proposed.distance,
                                                                         best_alns.vehicles, best_alns.distance),
            }
            comparison_rows.append(row)
            if instance.family not in representatives:
                representatives[instance.family] = (instance, baseline, copy.deepcopy(best_alns), proposed)
            checkpoint = {"completed_instance": instance.name, "completed_index": index,
                          "total_instances": len(instances), "created_at": datetime.now().isoformat(timespec="seconds")}
            (folders["checkpoints"] / "E5_progress.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

        alns_runs = pd.DataFrame(alns_run_rows)
        paired = pd.DataFrame(paired_rows)
        comparison = pd.DataFrame(comparison_rows)
        convergence = pd.DataFrame(convergence_rows)
        operators = pd.DataFrame(operator_rows)
        validations = pd.DataFrame(validation_rows)
        statistics = build_statistics(paired)
        family_summary = comparison.groupby("family", as_index=False).agg(
            instances=("instance", "count"),
            mean_bks_vehicles=("bks_vehicles", "mean"), mean_bih_vehicles=("bih_vehicles", "mean"),
            mean_alns_vehicles=("alns_vehicles", "mean"), mean_da_adr_vehicles=("da_adr_vehicles", "mean"),
            mean_bks_distance=("bks_distance", "mean"), mean_bih_distance=("bih_distance", "mean"),
            mean_alns_distance=("alns_distance", "mean"), mean_da_adr_distance=("da_adr_distance", "mean"),
            mean_alns_vehicle_gap_vs_bks=("alns_vehicle_gap_vs_bks", "mean"),
            mean_da_adr_vehicle_gap_vs_bks=("da_adr_vehicle_gap_vs_bks", "mean"),
            mean_da_adr_gain_vs_alns_percent=("da_adr_gain_vs_alns_percent", "mean"),
        )
        wins = comparison.groupby(["family", "da_adr_vs_alns_lexicographic"]).size().rename("instances").reset_index()
        files = {
            "E5_SINTEF_official_BKS_snapshot.csv": bks,
            "E5_standard_ALNS_all_runs.csv": alns_runs,
            "E5_paired_seed_comparison.csv": paired,
            "E5_instance_benchmark_comparison.csv": comparison,
            "E5_family_benchmark_summary.csv": family_summary,
            "E5_statistical_tests.csv": statistics,
            "E5_lexicographic_win_tie_loss.csv": wins,
            "E5_ALNS_convergence_history.csv": convergence,
            "E5_ALNS_operator_adaptation.csv": operators,
            "E5_independent_route_validation.csv": validations,
        }
        for name, frame in files.items():
            frame.to_csv(folders["tables"] / name, index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(args.output / "E5_External_Comparison_Publication_Ready.xlsx", engine="openpyxl") as writer:
            for title, frame in [("Family summary", family_summary), ("Instance comparison", comparison),
                                 ("Paired seeds", paired), ("Statistical tests", statistics),
                                 ("ALNS all runs", alns_runs), ("SINTEF BKS", bks),
                                 ("Route validation", validations), ("Win tie loss", wins)]:
                frame.to_excel(writer, sheet_name=title, index=False)
        create_charts(comparison, paired, convergence, e4_convergence, folders["charts"])
        for family in FAMILIES:
            if family in representatives:
                instance, baseline, alns, proposed = representatives[family]
                route_map(instance, baseline, alns, proposed, folders["maps"] / f"{instance.name}_BIH_ALNS_DA_ADR_routes")

        elapsed = time.perf_counter() - started
        lex = comparison.da_adr_vs_alns_lexicographic.value_counts().to_dict()
        manifest = {
            "experiment": "E5 - BIH, standard ALNS, DA-ADR and official SINTEF BKS",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "inputs": {"E1": str(args.e1), "E2": str(args.e2), "E3": str(args.e3),
                       "E4": str(args.e4), "dataset": str(args.dataset)},
            **lineage,
            "instances": len(instances), "runs_per_instance": args.runs,
            "matched_e4_iterations": args.iterations, "matched_e4_time_limit_seconds": args.time_limit,
            "base_seed": base_seed,
            "standard_alns_uses_e3_p4": False,
            "all_standard_alns_feasible": bool(alns_runs.feasible.all()),
            "all_independent_route_validations_passed": bool(validations.feasible.all()),
            "sintef_bks_source": BKS_SOURCE, "sintef_bks_access_date": BKS_ACCESSED,
            "bks_distance_comparison_note": "Distance gap has lexicographic meaning only when fleet sizes are equal.",
            "da_adr_vs_alns_lexicographic": lex,
            "statistical_method": "Wilcoxon signed-rank" if SCIPY_AVAILABLE else "Exact sign test",
            "multiple_testing_correction": "Holm-Bonferroni",
            "elapsed_seconds": elapsed,
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
                            "scipy_available": SCIPY_AVAILABLE},
        }
        (args.output / "E5_reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("=" * 78)
        logger.info("E5 COMPLETE | instances=%d | ALNS runs=%d | all feasible=%s", len(instances), len(alns_runs),
                    bool(alns_runs.feasible.all()))
        logger.info("DA-ADR vs ALNS lexicographic outcomes: %s", lex)
        logger.info("Elapsed %.1f minutes | Results=%s", elapsed / 60.0, args.output)
        logger.info("=" * 78)
        if platform.system() == "Windows":
            logger.info("E5 completed successfully. Windows will shut down in 60 seconds.")
            logger.info("To cancel shutdown, run: shutdown /a")
            try:
                subprocess.run(
                    ["shutdown", "/s", "/t", "60", "/c",
                     "E5 completed successfully. All experiment results have been saved."],
                    check=True,
                )
            except (OSError, subprocess.CalledProcessError):
                logger.exception("Results were saved, but automatic Windows shutdown could not be scheduled")
        else:
            logger.info("Automatic shutdown skipped: this computer is not running Windows.")
        return 0
    except Exception:
        logger.exception("E5 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
