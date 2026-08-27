# -*- coding: utf-8 -*-
"""E4: Difficulty-Aware Adaptive Destroy-and-Repair (DA-ADR).

Inputs are the E1, E2 and E3 ZIP archives produced by the preceding stages.
The program reconstructs every Homberger instance from E1, initializes the
search with the independently validated BIH solution from E2, and uses the P4
order-difficulty scores from E3 to guide adaptive destroy-and-repair search.

The E2 solution is permanently retained in the archive (non-degradation
safeguard).  Publication mode uses five independent seeds by default.  Use
--quick for a short pipeline check before launching the complete experiment.

Windows PowerShell:
    python E4_DA_ADR_Optimizer.py
    python E4_DA_ADR_Optimizer.py --quick
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
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = (
    "F:\\dulieu\\Nghien cứu sinh\\Nghiên cứu sinh chính thức\\"
    r"Paper_An Order-Priority Adaptive Heuristic for Large-Scale Last-Mile Delivery with Time Windows\Empirical"
)
DEFAULT_E1 = Path(ROOT + r"\E1.zip")
DEFAULT_E2 = Path(ROOT + r"\E2.zip")
DEFAULT_E3 = Path(ROOT + r"\E3.zip")
DEFAULT_OUTPUT = Path(ROOT + r"\E4")
FAMILIES = ("C1", "C2", "R1", "R2", "RC1", "RC2")
DESTROY_OPERATORS = ("difficulty", "related", "worst_arc", "random", "route_elimination")
EPS = 1e-8


@dataclass
class Instance:
    name: str
    family: str
    capacity: float
    x: np.ndarray
    y: np.ndarray
    demand: np.ndarray
    ready: np.ndarray
    due: np.ndarray
    service: np.ndarray
    distance: np.ndarray
    difficulty: np.ndarray


@dataclass
class Solution:
    routes: list[list[int]]
    vehicles: int
    distance: float


def natural_key(value: str) -> tuple:
    import re
    return tuple(int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_output(root: Path) -> dict[str, Path]:
    folders = {
        "root": root,
        "tables": root / "tables",
        "solutions": root / "best_solutions_json",
        "charts": root / "charts",
        "maps": root / "route_maps",
        "checkpoints": root / "checkpoints",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def setup_logger(root: Path) -> logging.Logger:
    logger = logging.getLogger("E4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(root / "E4_execution.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger


def zip_member(zf: zipfile.ZipFile, suffix: str) -> str:
    suffix = suffix.replace("\\", "/").lower()
    candidates = [n for n in zf.namelist() if n.replace("\\", "/").lower().endswith(suffix)]
    if not candidates:
        raise FileNotFoundError(f"Missing {suffix} in {zf.filename}")
    return min(candidates, key=len)


def read_json(zf: zipfile.ZipFile, suffix: str) -> dict:
    return json.loads(zf.read(zip_member(zf, suffix)).decode("utf-8-sig"))


def read_csv(zf: zipfile.ZipFile, suffix: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(zf.read(zip_member(zf, suffix))))


def validate_inputs(e1: Path, e2: Path, e3: Path) -> dict:
    for path in (e1, e2, e3):
        if not path.is_file():
            raise FileNotFoundError(f"Input ZIP not found: {path}")
    with zipfile.ZipFile(e1) as z1, zipfile.ZipFile(e2) as z2, zipfile.ZipFile(e3) as z3:
        m1 = read_json(z1, "E1_reproducibility_manifest.json")
        m2 = read_json(z2, "E2_reproducibility_manifest.json")
        m3 = read_json(z3, "E3_reproducibility_manifest.json")
    hashes = {
        "e1_dataset_sha256": m1.get("dataset_sha256"),
        "e2_dataset_sha256": m2.get("dataset_sha256"),
        "e3_dataset_sha256": m3.get("dataset_sha256"),
    }
    present = [v for v in hashes.values() if v]
    if len(present) != 3 or len(set(present)) != 1:
        raise ValueError(f"E1/E2/E3 dataset lineage mismatch: {hashes}")
    if int(m1.get("valid_instances", m1.get("instances", 0))) not in (0, 60):
        raise ValueError("E1 does not certify all 60 instances")
    return {"dataset_sha256": present[0], **hashes}


def load_data(e1_path: Path, e2_path: Path, e3_path: Path, families: set[str], limit: int) -> tuple[list[Instance], dict[str, dict]]:
    with zipfile.ZipFile(e1_path) as z1:
        customers = read_csv(z1, "E1_all_customer_characteristics.csv")
        summaries = read_csv(z1, "E1_instance_summary.csv").set_index("instance")
    with zipfile.ZipFile(e3_path) as z3:
        scores = read_csv(z3, "E3_all_order_difficulty_scores.csv")
    score_lookup = scores.set_index(["instance", "customer_id"])["P4_proposed"]
    solutions: dict[str, dict] = {}
    with zipfile.ZipFile(e2_path) as z2:
        names = sorted([n for n in z2.namelist() if n.lower().endswith("_best.json")], key=natural_key)
        for name in names:
            data = json.loads(z2.read(name).decode("utf-8-sig"))
            solutions[data["instance"]] = data

    instances: list[Instance] = []
    grouped = customers.groupby("instance", sort=False)
    names = sorted([n for n in grouped.groups if str(summaries.loc[n, "family"]) in families], key=natural_key)
    if limit:
        names = names[:limit]
    for name in names:
        frame = grouped.get_group(name).sort_values("customer_id")
        summary = summaries.loc[name]
        ids = frame["customer_id"].astype(int).to_numpy()
        if not np.array_equal(ids, np.arange(1, len(frame) + 1)):
            raise ValueError(f"Non-contiguous customer IDs in {name}")
        n = len(frame)
        x = np.r_[float(summary.depot_x), frame.x.to_numpy(float)]
        y = np.r_[float(summary.depot_y), frame.y.to_numpy(float)]
        demand = np.r_[0.0, frame.demand.to_numpy(float)]
        ready = np.r_[float(summary.depot_ready_time), frame.ready_time.to_numpy(float)]
        due = np.r_[float(summary.depot_due_date), frame.due_date.to_numpy(float)]
        service = np.r_[0.0, frame.service_time.to_numpy(float)]
        distance = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
        difficulty = np.zeros(n + 1)
        difficulty[1:] = [float(score_lookup.loc[(name, int(i))]) for i in ids]
        if name not in solutions:
            raise ValueError(f"Missing E2 best solution for {name}")
        instances.append(Instance(name, str(summary.family), float(summary.capacity), x, y, demand, ready, due, service, distance, difficulty))
    return instances, solutions


def route_metrics(instance: Instance, route: list[int]) -> tuple[bool, float, float]:
    load = 0.0
    clock = instance.ready[0]
    distance = 0.0
    for previous, node in zip(route, route[1:]):
        distance += instance.distance[previous, node]
        clock = max(instance.ready[node], clock + instance.service[previous] + instance.distance[previous, node])
        if clock > instance.due[node] + EPS:
            return False, distance, load
        if node != 0:
            load += instance.demand[node]
            if load > instance.capacity + EPS:
                return False, distance, load
    return True, distance, load


def evaluate(instance: Instance, routes: list[list[int]], require_coverage: bool = True) -> Solution:
    clean = [r for r in routes if len(r) > 2]
    total = 0.0
    served: list[int] = []
    for route in clean:
        if route[0] != 0 or route[-1] != 0:
            raise ValueError("Route must start and end at depot 0")
        feasible, distance, _ = route_metrics(instance, route)
        if not feasible:
            raise ValueError(f"Infeasible route in {instance.name}")
        total += distance
        served.extend(route[1:-1])
    if require_coverage:
        expected = list(range(1, len(instance.x)))
        if sorted(served) != expected:
            raise ValueError(f"Coverage/duplication failure in {instance.name}")
    return Solution(clean, len(clean), total)


def objective(solution: Solution) -> tuple[int, float]:
    return solution.vehicles, solution.distance


def better(a: Solution, b: Solution) -> bool:
    return a.vehicles < b.vehicles or (a.vehicles == b.vehicles and a.distance < b.distance - EPS)


def remove_customers(routes: list[list[int]], removed: set[int]) -> list[list[int]]:
    return [[0] + [node for node in route[1:-1] if node not in removed] + [0] for route in routes]


def select_destroy(instance: Instance, solution: Solution, operator: str, fraction: float, rng: random.Random) -> set[int]:
    customers = [node for route in solution.routes for node in route[1:-1]]
    q = max(8, min(len(customers) - 1, int(math.ceil(fraction * len(customers)))))
    if operator == "route_elimination":
        candidates = sorted(solution.routes, key=lambda r: (len(r), sum(instance.difficulty[r[1:-1]])))
        pool: list[int] = []
        for route in candidates[: max(1, min(4, len(candidates)))]:
            pool.extend(route[1:-1])
        chosen_route = rng.choice(candidates[: max(1, min(4, len(candidates)))])
        return set(chosen_route[1:-1])
    if operator == "difficulty":
        weights = np.asarray([0.05 + instance.difficulty[c] ** 2 for c in customers], dtype=float)
        keys = [rng.random() ** (1.0 / max(w, 1e-9)) for w in weights]
        return set(c for _, c in sorted(zip(keys, customers), reverse=True)[:q])
    if operator == "related":
        seed = rng.choice(customers)
        max_d = max(float(instance.distance[seed].max()), 1.0)
        max_t = max(float(instance.due.max() - instance.ready.min()), 1.0)
        related = sorted(
            customers,
            key=lambda c: (
                instance.distance[seed, c] / max_d
                + 0.35 * abs(instance.ready[seed] - instance.ready[c]) / max_t
                + 0.20 * abs(instance.difficulty[seed] - instance.difficulty[c])
                + rng.random() * 0.05
            ),
        )
        return set(related[:q])
    if operator == "worst_arc":
        savings = []
        for route in solution.routes:
            for pos in range(1, len(route) - 1):
                a, c, b = route[pos - 1], route[pos], route[pos + 1]
                saving = instance.distance[a, c] + instance.distance[c, b] - instance.distance[a, b]
                savings.append((saving * (0.8 + 0.4 * rng.random()), c))
        return set(c for _, c in sorted(savings, reverse=True)[:q])
    return set(rng.sample(customers, q))


def candidate_routes(instance: Instance, routes: list[list[int]], customer: int, maximum: int = 10) -> list[int]:
    if len(routes) <= maximum:
        return list(range(len(routes)))
    ranked = []
    for index, route in enumerate(routes):
        nodes = route[1:-1]
        proximity = min((instance.distance[customer, node] for node in nodes), default=instance.distance[customer, 0])
        ranked.append((proximity, index))
    return [index for _, index in sorted(ranked)[:maximum]]


def insertion_options(instance: Instance, routes: list[list[int]], customer: int) -> list[tuple[float, int, int]]:
    options: list[tuple[float, int, int]] = []
    for route_index in candidate_routes(instance, routes, customer):
        route = routes[route_index]
        for position in range(1, len(route)):
            previous, following = route[position - 1], route[position]
            delta = (
                instance.distance[previous, customer]
                + instance.distance[customer, following]
                - instance.distance[previous, following]
            )
            trial = route[:position] + [customer] + route[position:]
            feasible, _, _ = route_metrics(instance, trial)
            if feasible:
                options.append((float(delta), route_index, position))
    options.sort()
    return options


def repair(instance: Instance, partial: list[list[int]], removed: set[int], rng: random.Random, mode: str) -> list[list[int]] | None:
    routes = [r for r in partial if len(r) > 2]
    pending = list(removed)
    while pending:
        candidates = []
        # A restricted candidate list is essential at the 1,000-customer scale.
        # It preserves adaptive regret behavior while avoiding an exhaustive
        # O(removed^2 * routes * positions) repair at every iteration.
        inspect = pending if len(pending) <= 8 else sorted(pending, key=lambda c: instance.difficulty[c], reverse=True)[:8]
        for customer in inspect:
            options = insertion_options(instance, routes, customer)
            if options:
                regret = (options[1][0] - options[0][0]) if len(options) > 1 else max(1.0, options[0][0])
                priority = instance.difficulty[customer]
                if mode == "priority":
                    key = priority + 0.02 * rng.random()
                elif mode == "regret2":
                    key = regret + 0.001 * priority
                else:
                    key = regret * (0.50 + priority) + 0.05 * priority + 0.001 * rng.random()
                candidates.append((key, customer, options[0]))
            else:
                singleton = [0, customer, 0]
                feasible, distance, _ = route_metrics(instance, singleton)
                if not feasible:
                    return None
                candidates.append((-1e12, customer, (distance, len(routes), 1)))
        if not candidates:
            return None
        _, customer, (_, route_index, position) = max(candidates, key=lambda row: row[0])
        if route_index == len(routes):
            routes.append([0, customer, 0])
        else:
            routes[route_index] = routes[route_index][:position] + [customer] + routes[route_index][position:]
        pending.remove(customer)
    return routes


def choose_weighted(weights: dict[str, float], rng: random.Random) -> str:
    names = list(weights)
    return rng.choices(names, weights=[weights[n] for n in names], k=1)[0]


def optimize(instance: Instance, baseline: Solution, seed: int, iterations: int, time_limit: float, destroy_min: float, destroy_max: float) -> tuple[Solution, list[dict], list[dict]]:
    rng = random.Random(seed)
    current = copy.deepcopy(baseline)
    best = copy.deepcopy(baseline)  # permanent E2 archive member / safeguard
    weights = {name: 1.0 for name in DESTROY_OPERATORS}
    uses = {name: 0 for name in DESTROY_OPERATORS}
    rewards = {name: 0.0 for name in DESTROY_OPERATORS}
    history: list[dict] = []
    started = time.perf_counter()
    temperature = max(1.0, 0.0025 * baseline.distance)
    stagnant = 0
    completed = 0
    for iteration in range(1, iterations + 1):
        if time.perf_counter() - started >= time_limit:
            break
        operator = choose_weighted(weights, rng)
        mode = rng.choices(["hybrid", "priority", "regret2"], weights=[0.55, 0.25, 0.20], k=1)[0]
        fraction = rng.uniform(destroy_min, destroy_max) * (1.35 if stagnant > 35 else 1.0)
        fraction = min(0.25, fraction)
        removed = select_destroy(instance, current, operator, fraction, rng)
        partial = remove_customers(current.routes, removed)
        rebuilt = repair(instance, partial, removed, rng, mode)
        uses[operator] += 1
        reward = 0.0
        accepted = False
        if rebuilt is not None:
            try:
                candidate = evaluate(instance, rebuilt)
            except ValueError:
                candidate = None
            if candidate is not None:
                if better(candidate, best):
                    best = copy.deepcopy(candidate)
                    current = candidate
                    reward = 8.0
                    accepted = True
                    stagnant = 0
                elif better(candidate, current):
                    current = candidate
                    reward = 4.0
                    accepted = True
                    stagnant += 1
                elif candidate.vehicles == current.vehicles:
                    probability = math.exp(-max(0.0, candidate.distance - current.distance) / max(temperature, 1e-9))
                    if rng.random() < probability:
                        current = candidate
                        reward = 1.0
                        accepted = True
                    stagnant += 1
                else:
                    stagnant += 1
        else:
            stagnant += 1
        rewards[operator] += reward
        weights[operator] = max(0.10, 0.80 * weights[operator] + 0.20 * (1.0 + reward))
        temperature *= 0.992
        completed = iteration
        history.append({
            "iteration": iteration,
            "elapsed_seconds": time.perf_counter() - started,
            "operator": operator,
            "repair": mode,
            "removed": len(removed),
            "accepted": accepted,
            "current_vehicles": current.vehicles,
            "current_distance": current.distance,
            "best_vehicles": best.vehicles,
            "best_distance": best.distance,
        })
    operator_rows = [
        {"operator": name, "uses": uses[name], "total_reward": rewards[name], "final_weight": weights[name]}
        for name in DESTROY_OPERATORS
    ]
    if better(baseline, best):
        raise AssertionError("Non-degradation safeguard failed")
    if completed == 0:
        history.append({"iteration": 0, "elapsed_seconds": 0.0, "operator": "none", "repair": "none", "removed": 0,
                        "accepted": False, "current_vehicles": baseline.vehicles, "current_distance": baseline.distance,
                        "best_vehicles": baseline.vehicles, "best_distance": baseline.distance})
    return best, history, operator_rows


def save_solution(path: Path, instance: Instance, solution: Solution, baseline: Solution, seed: int, feasible: bool) -> None:
    payload = {
        "instance": instance.name,
        "family": instance.family,
        "algorithm": "DA-ADR",
        "seed": seed,
        "objective": {"vehicle_count": solution.vehicles, "total_distance": solution.distance},
        "e2_baseline": {"vehicle_count": baseline.vehicles, "total_distance": baseline.distance},
        "non_degradation_verified": not better(baseline, solution),
        "feasible": feasible,
        "routes": solution.routes,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_plot(fig, folder: Path, name: str) -> None:
    fig.savefig(folder / f"{name}.png", dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_charts(runs: pd.DataFrame, convergence: pd.DataFrame, family: pd.DataFrame, charts: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(len(FAMILIES))
    frame = family.set_index("family").reindex(FAMILIES)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(x, frame["mean_distance_improvement_percent"], color="#2369a1")
    ax.set_xticks(x, FAMILIES); ax.set_ylabel("Mean distance improvement (%)")
    ax.set_title("DA-ADR improvement over the retained E2 baseline")
    ax.axhline(0, color="black", linewidth=0.8)
    save_plot(fig, charts, "01_distance_improvement_by_family")

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(x, frame["mean_vehicle_reduction"], color="#159d82")
    ax.set_xticks(x, FAMILIES); ax.set_ylabel("Mean vehicle reduction")
    ax.set_title("Fleet-size reduction relative to E2")
    ax.axhline(0, color="black", linewidth=0.8)
    save_plot(fig, charts, "02_vehicle_reduction_by_family")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for family_name in FAMILIES:
        subset = convergence[convergence.family == family_name]
        if subset.empty: continue
        normalized = []
        for _, group in subset.groupby(["instance", "seed"]):
            base = float(group.iloc[0].baseline_distance)
            normalized.append(pd.Series(100 * (base - group.best_distance.to_numpy()) / base,
                                        index=np.linspace(0, 1, len(group))))
        matrix = pd.concat(normalized, axis=1).interpolate().ffill()
        ax.plot(matrix.index, matrix.mean(axis=1), label=family_name)
    ax.set_xlabel("Normalized search progress"); ax.set_ylabel("Distance improvement (%)")
    ax.set_title("Convergence of DA-ADR"); ax.legend(ncol=3)
    save_plot(fig, charts, "03_convergence_by_family")

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    data = [runs.loc[runs.family == f, "distance_improvement_percent"].to_numpy() for f in FAMILIES]
    ax.boxplot(data, tick_labels=FAMILIES, showmeans=True)
    ax.set_ylabel("Distance improvement (%)"); ax.set_title("Run-level robustness across instance families")
    save_plot(fig, charts, "04_run_robustness_boxplot")


def route_map(instance: Instance, baseline: Solution, best: Solution, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, solution, title in zip(axes, [baseline, best], ["E2 BIH baseline", "E4 DA-ADR"]):
        cmap = plt.get_cmap("tab20")
        for idx, route in enumerate(solution.routes):
            nodes = np.asarray(route, dtype=int)
            ax.plot(instance.x[nodes], instance.y[nodes], color=cmap(idx % 20), linewidth=0.65, alpha=0.75)
        ax.scatter(instance.x[1:], instance.y[1:], s=5, c=instance.difficulty[1:], cmap="magma", alpha=0.75)
        ax.scatter(instance.x[0], instance.y[0], marker="*", s=170, color="#00a878", edgecolor="black", zorder=5)
        ax.set_title(f"{title}\nvehicles={solution.vehicles}, distance={solution.distance:.1f}")
        ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("X"); ax.grid(alpha=0.15)
    axes[0].set_ylabel("Y")
    fig.suptitle(f"{instance.name}: retained baseline versus optimized delivery routes")
    save_plot(fig, path.parent, path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E4 DA-ADR optimizer for Homberger1000")
    parser.add_argument("--e1", type=Path, default=DEFAULT_E1)
    parser.add_argument("--e2", type=Path, default=DEFAULT_E2)
    parser.add_argument("--e3", type=Path, default=DEFAULT_E3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=int, default=5, help="Independent seeds per instance")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--time-limit", type=float, default=45.0, help="Seconds per instance and seed")
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--destroy-min", type=float, default=0.02)
    parser.add_argument("--destroy-max", type=float, default=0.06)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--limit-instances", type=int, default=0, help="Testing only; 0 means all")
    parser.add_argument("--quick", action="store_true", help="Pipeline test: one instance, one run, 10 iterations, 15 seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.limit_instances, args.runs, args.iterations, args.time_limit = 1, 1, 10, 15.0
        args.output = args.output.parent / (args.output.name + "_quick_test")
    folders = setup_output(args.output)
    logger = setup_logger(args.output)
    started = time.perf_counter()
    logger.info("Starting E4 DA-ADR")
    logger.info("E1=%s", args.e1); logger.info("E2=%s", args.e2); logger.info("E3=%s", args.e3)
    logger.info("Output=%s", args.output)
    logger.info("Mode: runs=%d, iterations=%d, time-limit=%.1fs/run", args.runs, args.iterations, args.time_limit)
    try:
        lineage = validate_inputs(args.e1, args.e2, args.e3)
        instances, e2_solutions = load_data(args.e1, args.e2, args.e3, set(args.families), args.limit_instances)
        logger.info("Lineage verified; loaded %d instances", len(instances))
        run_rows, convergence_rows, operator_rows = [], [], []
        best_by_instance: dict[str, tuple[Instance, Solution, Solution, int]] = {}
        for index, instance in enumerate(instances, 1):
            raw = e2_solutions[instance.name]
            baseline = evaluate(instance, [[int(n) for n in route] for route in raw["routes"]])
            overall_best = copy.deepcopy(baseline)
            overall_seed = args.base_seed
            for run in range(args.runs):
                seed = args.base_seed + run
                run_start = time.perf_counter()
                best, history, ops = optimize(instance, baseline, seed, args.iterations, args.time_limit, args.destroy_min, args.destroy_max)
                elapsed = time.perf_counter() - run_start
                verified = evaluate(instance, best.routes)
                if objective(verified) != objective(best):
                    raise AssertionError("Independent objective verification failed")
                if better(best, overall_best):
                    overall_best, overall_seed = copy.deepcopy(best), seed
                vehicle_reduction = baseline.vehicles - best.vehicles
                distance_gain = 100.0 * (baseline.distance - best.distance) / baseline.distance
                run_rows.append({
                    "instance": instance.name, "family": instance.family, "run": run + 1, "seed": seed,
                    "baseline_vehicles": baseline.vehicles, "baseline_distance": baseline.distance,
                    "best_vehicles": best.vehicles, "best_distance": best.distance,
                    "vehicle_reduction": vehicle_reduction, "distance_improvement_percent": distance_gain,
                    "feasible": True, "non_degradation": not better(baseline, best),
                    "iterations_completed": history[-1]["iteration"], "runtime_seconds": elapsed,
                })
                for row in history:
                    convergence_rows.append({"instance": instance.name, "family": instance.family, "run": run + 1,
                                             "seed": seed, "baseline_distance": baseline.distance, **row})
                for row in ops:
                    operator_rows.append({"instance": instance.name, "family": instance.family, "run": run + 1,
                                          "seed": seed, **row})
                logger.info("[%02d/%02d] %-11s run=%d/%d seed=%d | %d -> %d vehicles | distance %.1f -> %.1f (%+.2f%%) | %.1fs",
                            index, len(instances), instance.name, run + 1, args.runs, seed,
                            baseline.vehicles, best.vehicles, baseline.distance, best.distance, distance_gain, elapsed)
            best_by_instance[instance.name] = (instance, baseline, overall_best, overall_seed)
            save_solution(folders["solutions"] / f"{instance.name}_DA_ADR_best.json", instance, overall_best, baseline, overall_seed, True)
            checkpoint = {"completed_instance": instance.name, "completed_index": index, "total_instances": len(instances),
                          "created_at": datetime.now().isoformat(timespec="seconds")}
            (folders["checkpoints"] / "E4_progress.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

        runs = pd.DataFrame(run_rows)
        convergence = pd.DataFrame(convergence_rows)
        operators = pd.DataFrame(operator_rows)
        best_runs = runs.sort_values(["instance", "best_vehicles", "best_distance"]).groupby("instance", as_index=False).first()
        family = best_runs.groupby("family", as_index=False).agg(
            instances=("instance", "count"), feasible_rate=("feasible", "mean"),
            non_degradation_rate=("non_degradation", "mean"), mean_baseline_vehicles=("baseline_vehicles", "mean"),
            mean_best_vehicles=("best_vehicles", "mean"), mean_vehicle_reduction=("vehicle_reduction", "mean"),
            mean_baseline_distance=("baseline_distance", "mean"), mean_best_distance=("best_distance", "mean"),
            mean_distance_improvement_percent=("distance_improvement_percent", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        runs.to_csv(folders["tables"] / "E4_all_runs.csv", index=False, encoding="utf-8-sig")
        best_runs.to_csv(folders["tables"] / "E4_best_results_by_instance.csv", index=False, encoding="utf-8-sig")
        family.to_csv(folders["tables"] / "E4_family_summary.csv", index=False, encoding="utf-8-sig")
        convergence.to_csv(folders["tables"] / "E4_convergence_history.csv", index=False, encoding="utf-8-sig")
        operators.to_csv(folders["tables"] / "E4_operator_adaptation.csv", index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(args.output / "E4_DA_ADR_Publication_Ready.xlsx", engine="openpyxl") as writer:
            family.to_excel(writer, sheet_name="Family summary", index=False)
            best_runs.to_excel(writer, sheet_name="Best by instance", index=False)
            runs.to_excel(writer, sheet_name="All runs", index=False)
            operators.to_excel(writer, sheet_name="Operator adaptation", index=False)
        create_charts(runs, convergence, family, folders["charts"])
        representatives = {}
        for fam in FAMILIES:
            names = sorted([n for n, values in best_by_instance.items() if values[0].family == fam], key=natural_key)
            if names: representatives[fam] = names[0]
        for name in representatives.values():
            instance, baseline, best, _ = best_by_instance[name]
            route_map(instance, baseline, best, folders["maps"] / f"{name}_E2_vs_E4_routes")
        elapsed = time.perf_counter() - started
        manifest = {
            "experiment": "E4 - Complete DA-ADR", "created_at": datetime.now().isoformat(timespec="seconds"),
            "e1_zip": str(args.e1), "e2_zip": str(args.e2), "e3_zip": str(args.e3),
            "e1_zip_sha256": sha256_file(args.e1), "e2_zip_sha256": sha256_file(args.e2), "e3_zip_sha256": sha256_file(args.e3),
            **lineage, "instances": len(instances), "runs_per_instance": args.runs, "iterations": args.iterations,
            "time_limit_seconds_per_run": args.time_limit, "base_seed": args.base_seed,
            "non_degradation_safeguard": "E2 solution retained in archive and independently revalidated",
            "all_feasible": bool(runs.feasible.all()), "all_non_degraded": bool(runs.non_degradation.all()),
            "elapsed_seconds": elapsed,
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        }
        (args.output / "E4_reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("=" * 78)
        logger.info("E4 COMPLETE | instances=%d | runs=%d | feasible=100%% | non-degradation=100%%", len(instances), len(runs))
        logger.info("Elapsed %.1f minutes | results=%s", elapsed / 60.0, args.output)
        logger.info("=" * 78)
        return 0
    except Exception:
        logger.exception("E4 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
