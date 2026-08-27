#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2: Multi-strategy Best-Insertion Heuristic (BIH) baseline.

The program reads the original Homberger1000 ZIP and the E1 ZIP, verifies
dataset lineage using SHA-256, runs four deterministic BIH strategies on all
60 instances, validates every route independently, and exports publication-
ready tables, charts, JSON solutions, and representative delivery-route maps.

Windows quick start:
    python E2_Homberger1000_BIH_Baseline.py

Dependencies:
    python -m pip install numpy pandas matplotlib openpyxl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    print("Missing package:", exc)
    print("Install with: python -m pip install numpy pandas matplotlib openpyxl")
    raise SystemExit(1) from exc


ROOT = (
    r"F:\dulieu\Nghien cứu sinh\Nghiên cứu sinh chính thức"
    r"\Paper_An Order-Priority Adaptive Heuristic for Large-Scale Last-Mile Delivery with Time Windows"
    r"\Empirical"
)
DEFAULT_DATASET = Path(ROOT + r"\homberger_1000_customer_instances.zip")
DEFAULT_E1 = Path(ROOT + r"\E1.zip")
DEFAULT_OUTPUT = Path(ROOT + r"\E2")

FAMILIES = ("C1", "C2", "R1", "R2", "RC1", "RC2")
STRATEGIES = ("earliest_due", "earliest_ready", "largest_demand", "farthest_from_depot")
STRATEGY_LABELS = {
    "earliest_due": "Earliest due",
    "earliest_ready": "Earliest ready",
    "largest_demand": "Largest demand",
    "farthest_from_depot": "Farthest from depot",
}
COLORS = {
    "earliest_due": "#2B6CB0",
    "earliest_ready": "#38A169",
    "largest_demand": "#DD6B20",
    "farthest_from_depot": "#805AD5",
}
EPS = 1e-8


@dataclass
class Instance:
    name: str
    family: str
    vehicle_limit: int
    capacity: float
    customer_id: np.ndarray
    x: np.ndarray
    y: np.ndarray
    demand: np.ndarray
    ready: np.ndarray
    due: np.ndarray
    service: np.ndarray
    distance: np.ndarray


@dataclass
class RouteEvaluation:
    feasible: bool
    distance: float
    load: float
    earliest: np.ndarray
    latest: np.ndarray
    waiting: np.ndarray
    cumulative_load: np.ndarray
    violations: list[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def natural_key(value: str) -> tuple:
    stem = Path(value).stem.upper()
    match = re.match(r"(RC1|RC2|C1|C2|R1|R2)", stem)
    family = match.group(1) if match else "ZZ"
    family_order = FAMILIES.index(family) if family in FAMILIES else 999
    return (family_order, tuple(int(v) for v in re.findall(r"\d+", stem)), stem)


def setup_output(output: Path) -> dict[str, Path]:
    folders = {
        "root": output,
        "tables": output / "tables",
        "charts": output / "charts",
        "routes": output / "route_maps",
        "solutions": output / "best_solutions_json",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def setup_logger(output: Path) -> logging.Logger:
    logger = logging.getLogger("homberger_e2")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(output / "E2_execution.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def read_e1_manifest(e1_path: Path) -> dict:
    if not e1_path.exists():
        raise FileNotFoundError(f"E1 ZIP not found: {e1_path}")
    if not zipfile.is_zipfile(e1_path):
        raise ValueError(f"E1 input is not a ZIP archive: {e1_path}")
    with zipfile.ZipFile(e1_path) as archive:
        candidates = [name for name in archive.namelist() if name.endswith("E1_reproducibility_manifest.json")]
        if len(candidates) != 1:
            raise ValueError(f"Expected one E1 manifest; found {len(candidates)}")
        return json.loads(archive.read(candidates[0]).decode("utf-8-sig"))


def verify_lineage(dataset_path: Path, e1_manifest: dict, allow_mismatch: bool) -> dict:
    dataset_hash = sha256_file(dataset_path)
    e1_hash = e1_manifest.get("dataset_sha256")
    match = bool(e1_hash) and dataset_hash.lower() == str(e1_hash).lower()
    if not match and not allow_mismatch:
        raise ValueError(
            "The Homberger1000 ZIP does not match the dataset verified in E1.\n"
            f"Current SHA-256: {dataset_hash}\nE1 SHA-256: {e1_hash}\n"
            "Use the same dataset, or use --allow-hash-mismatch only after manual verification."
        )
    if int(e1_manifest.get("valid_instances", 0)) != int(e1_manifest.get("instances", -1)):
        raise ValueError("E1 did not validate every instance; E2 has been stopped.")
    return {"dataset_sha256": dataset_hash, "e1_dataset_sha256": e1_hash, "hash_match": match}


def parse_instance(name: str, payload: bytes) -> Instance:
    text = payload.decode("utf-8-sig", errors="replace")
    lines = [line.strip() for line in text.splitlines()]
    nonempty = [line for line in lines if line]
    instance_name = nonempty[0].upper()
    family_match = re.match(r"(RC1|RC2|C1|C2|R1|R2)", instance_name)
    if not family_match:
        raise ValueError(f"Unknown family in {name}")
    family = family_match.group(1)
    vehicle_header = next(i for i, line in enumerate(lines) if "NUMBER" in line.upper() and "CAPACITY" in line.upper())
    customer_header = next(i for i, line in enumerate(lines) if "CUST" in line.upper() and "DEMAND" in line.upper())
    vehicle_line = next(line for line in lines[vehicle_header + 1 :] if line)
    vehicle_values = [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", vehicle_line)]
    vehicle_limit, capacity = int(vehicle_values[0]), float(vehicle_values[1])
    records = []
    for line in lines[customer_header + 1 :]:
        if line:
            values = line.split()
            if len(values) >= 7:
                records.append([float(value) for value in values[:7]])
    data = np.asarray(records, dtype=np.float64)
    if data.shape != (1001, 7):
        raise ValueError(f"{instance_name}: expected 1001 rows x 7 columns; found {data.shape}")
    ids = data[:, 0].astype(int)
    order = np.argsort(ids)
    data, ids = data[order], ids[order]
    if ids[0] != 0 or not np.array_equal(ids, np.arange(1001)):
        raise ValueError(f"{instance_name}: customer identifiers must be 0..1000")
    x, y = data[:, 1], data[:, 2]
    distance = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    return Instance(
        instance_name,
        family,
        vehicle_limit,
        capacity,
        ids,
        x,
        y,
        data[:, 3],
        data[:, 4],
        data[:, 5],
        data[:, 6],
        distance,
    )


def load_instances(dataset: Path, families: set[str], limit: int) -> list[Instance]:
    if not dataset.exists() or not zipfile.is_zipfile(dataset):
        raise FileNotFoundError(f"Dataset ZIP not found or invalid: {dataset}")
    instances: list[Instance] = []
    with zipfile.ZipFile(dataset) as archive:
        members = sorted(
            [name for name in archive.namelist() if name.lower().endswith(".txt")],
            key=natural_key,
        )
        for member in members:
            family_match = re.match(r"(RC1|RC2|C1|C2|R1|R2)", Path(member).stem.upper())
            if family_match and family_match.group(1) in families:
                instances.append(parse_instance(member, archive.read(member)))
                if limit > 0 and len(instances) >= limit:
                    break
    if not instances:
        raise ValueError("No matching Homberger instances were loaded.")
    return instances


def route_evaluation(instance: Instance, route: list[int]) -> RouteEvaluation:
    nodes = np.asarray(route, dtype=int)
    size = len(nodes)
    earliest = np.empty(size, dtype=float)
    latest = np.empty(size, dtype=float)
    waiting = np.zeros(size, dtype=float)
    cumulative = np.zeros(size, dtype=float)
    violations: list[str] = []
    earliest[0] = max(0.0, instance.ready[nodes[0]])
    for position in range(1, size):
        previous, node = nodes[position - 1], nodes[position]
        raw_arrival = earliest[position - 1] + instance.service[previous] + instance.distance[previous, node]
        earliest[position] = max(instance.ready[node], raw_arrival)
        waiting[position] = max(0.0, instance.ready[node] - raw_arrival)
        cumulative[position] = cumulative[position - 1] + (instance.demand[node] if node != 0 else 0.0)
        if earliest[position] > instance.due[node] + EPS:
            violations.append(f"time_window@{node}")
    latest[-1] = instance.due[nodes[-1]]
    for position in range(size - 2, -1, -1):
        node, following = nodes[position], nodes[position + 1]
        latest[position] = min(
            instance.due[node],
            latest[position + 1] - instance.service[node] - instance.distance[node, following],
        )
    load = float(instance.demand[nodes[(nodes != 0)]].sum())
    if load > instance.capacity + EPS:
        violations.append("capacity")
    if route[0] != 0 or route[-1] != 0:
        violations.append("depot_endpoints")
    length = float(instance.distance[nodes[:-1], nodes[1:]].sum())
    return RouteEvaluation(not violations, length, load, earliest, latest, waiting, cumulative, violations)


def customer_order(instance: Instance, strategy: str) -> list[int]:
    customers = np.arange(1, len(instance.customer_id))
    if strategy == "earliest_due":
        keys = (instance.distance[0, customers], instance.ready[customers], instance.due[customers])
    elif strategy == "earliest_ready":
        keys = (instance.due[customers], instance.distance[0, customers], instance.ready[customers])
    elif strategy == "largest_demand":
        keys = (instance.due[customers], instance.distance[0, customers], -instance.demand[customers])
    elif strategy == "farthest_from_depot":
        keys = (instance.due[customers], -instance.demand[customers], -instance.distance[0, customers])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return customers[np.lexsort(keys)].tolist()


def insertion_profile(instance: Instance, route: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
    evaluation = route_evaluation(instance, route)
    if not evaluation.feasible:
        raise ValueError("Internal error: insertion profile requested for infeasible route")
    return evaluation.earliest, evaluation.latest, evaluation.load


def best_insertion(instance: Instance, routes: list[list[int]], customer: int) -> tuple[int, int] | None:
    best_key: tuple[float, float, float, int, int] | None = None
    best_choice: tuple[int, int] | None = None
    customer_demand = instance.demand[customer]
    for route_index, route in enumerate(routes):
        earliest, latest, route_load = insertion_profile(instance, route)
        if route_load + customer_demand > instance.capacity + EPS:
            continue
        for position in range(1, len(route)):
            previous, following = route[position - 1], route[position]
            raw_arrival = earliest[position - 1] + instance.service[previous] + instance.distance[previous, customer]
            service_start = max(instance.ready[customer], raw_arrival)
            if service_start > instance.due[customer] + EPS:
                continue
            following_start = max(
                instance.ready[following],
                service_start + instance.service[customer] + instance.distance[customer, following],
            )
            if following_start > latest[position] + EPS:
                continue
            delta = (
                instance.distance[previous, customer]
                + instance.distance[customer, following]
                - instance.distance[previous, following]
            )
            wait = max(0.0, instance.ready[customer] - raw_arrival)
            slack = instance.due[customer] - service_start
            key = (float(delta), float(wait), float(-slack), route_index, position)
            if best_key is None or key < best_key:
                best_key, best_choice = key, (route_index, position)
    return best_choice


def construct_bih(instance: Instance, strategy: str) -> tuple[list[list[int]], dict]:
    started = time.perf_counter()
    routes: list[list[int]] = []
    new_route_count = 0
    for customer in customer_order(instance, strategy):
        choice = best_insertion(instance, routes, customer)
        if choice is None:
            route = [0, customer, 0]
            evaluation = route_evaluation(instance, route)
            if not evaluation.feasible:
                raise ValueError(f"{instance.name}: customer {customer} cannot form a feasible isolated route")
            routes.append(route)
            new_route_count += 1
            if len(routes) > instance.vehicle_limit:
                raise ValueError(f"{instance.name}: vehicle limit exceeded")
        else:
            route_index, position = choice
            routes[route_index].insert(position, customer)
    runtime = time.perf_counter() - started
    validation = validate_solution(instance, routes)
    validation.update({"runtime_seconds": runtime, "new_routes_created": new_route_count})
    return routes, validation


def validate_solution(instance: Instance, routes: list[list[int]]) -> dict:
    visits = [node for route in routes for node in route if node != 0]
    duplicate_customers = len(visits) - len(set(visits))
    missing_customers = len(set(range(1, 1001)) - set(visits))
    route_evaluations = [route_evaluation(instance, route) for route in routes]
    infeasible_routes = sum(not evaluation.feasible for evaluation in route_evaluations)
    total_distance = sum(evaluation.distance for evaluation in route_evaluations)
    total_waiting = sum(float(evaluation.waiting.sum()) for evaluation in route_evaluations)
    total_load = sum(evaluation.load for evaluation in route_evaluations)
    capacity_lower_bound = math.ceil(float(instance.demand[1:].sum()) / instance.capacity)
    feasible = duplicate_customers == 0 and missing_customers == 0 and infeasible_routes == 0 and len(routes) <= instance.vehicle_limit
    return {
        "feasible": feasible,
        "vehicle_count": len(routes),
        "total_distance": float(total_distance),
        "total_waiting": float(total_waiting),
        "total_load": float(total_load),
        "capacity_vehicle_lower_bound": capacity_lower_bound,
        "vehicle_excess_over_capacity_lb": len(routes) - capacity_lower_bound,
        "duplicate_customers": duplicate_customers,
        "missing_customers": missing_customers,
        "infeasible_routes": infeasible_routes,
    }


def solution_rows(instance: Instance, strategy: str, routes: list[list[int]]) -> tuple[list[dict], list[dict]]:
    route_rows: list[dict] = []
    stop_rows: list[dict] = []
    for route_id, route in enumerate(routes, start=1):
        evaluation = route_evaluation(instance, route)
        nodes = np.asarray(route, dtype=int)
        route_rows.append({
            "instance": instance.name,
            "family": instance.family,
            "strategy": strategy,
            "route_id": route_id,
            "customers": len(route) - 2,
            "load": evaluation.load,
            "capacity": instance.capacity,
            "capacity_utilization": evaluation.load / instance.capacity,
            "distance": evaluation.distance,
            "total_waiting": float(evaluation.waiting.sum()),
            "departure_time": float(evaluation.earliest[0]),
            "return_time": float(evaluation.earliest[-1]),
            "minimum_time_slack": float(np.min(evaluation.latest - evaluation.earliest)),
            "feasible": evaluation.feasible,
        })
        for sequence, node in enumerate(route):
            previous = route[sequence - 1] if sequence > 0 else None
            travel = instance.distance[previous, node] if previous is not None else 0.0
            raw_arrival = evaluation.earliest[sequence] - evaluation.waiting[sequence]
            stop_rows.append({
                "instance": instance.name,
                "family": instance.family,
                "strategy": strategy,
                "route_id": route_id,
                "sequence": sequence,
                "customer_id": node,
                "x": instance.x[node],
                "y": instance.y[node],
                "demand": instance.demand[node],
                "ready_time": instance.ready[node],
                "due_date": instance.due[node],
                "service_time": instance.service[node],
                "travel_from_previous": travel,
                "raw_arrival": raw_arrival,
                "service_start": evaluation.earliest[sequence],
                "waiting": evaluation.waiting[sequence],
                "latest_feasible_start": evaluation.latest[sequence],
                "cumulative_load": evaluation.cumulative_load[sequence],
            })
    return route_rows, stop_rows


def choose_best(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = summary.sort_values(
        ["instance", "vehicle_count", "total_distance", "runtime_seconds", "strategy"],
        kind="stable",
    )
    return ordered.groupby("instance", as_index=False).first()


def save_solution_json(instance: Instance, strategy: str, routes: list[list[int]], metrics: dict, folder: Path) -> None:
    payload = {
        "instance": instance.name,
        "family": instance.family,
        "strategy": strategy,
        "objective": {"vehicle_count": metrics["vehicle_count"], "total_distance": metrics["total_distance"]},
        "feasible": bool(metrics["feasible"]),
        "routes": routes,
    }
    (folder / f"{instance.name}_best.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_route_map(instance: Instance, routes: list[list[int]], strategy: str, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    cmap = plt.get_cmap("turbo", max(2, len(routes)))
    for index, route in enumerate(routes):
        nodes = np.asarray(route, dtype=int)
        color = cmap(index)
        axis.plot(instance.x[nodes], instance.y[nodes], color=color, linewidth=0.75, alpha=0.72)
        axis.scatter(instance.x[nodes[1:-1]], instance.y[nodes[1:-1]], s=8, color=color, alpha=0.78)
    axis.scatter(instance.x[0], instance.y[0], marker="*", s=260, color="#D7191C", edgecolors="black", zorder=10, label="Depot")
    total_distance = sum(route_evaluation(instance, route).distance for route in routes)
    axis.set_title(
        f"{instance.name} | {STRATEGY_LABELS[strategy]} | {len(routes)} vehicles | distance {total_distance:.2f}",
        fontweight="bold",
    )
    axis.set_xlabel("Synthetic X coordinate")
    axis.set_ylabel("Synthetic Y coordinate")
    axis.legend(loc="upper right")
    axis.set_aspect("equal", adjustable="box")
    figure.savefig(output_path.with_suffix(".png"), dpi=260, bbox_inches="tight", facecolor="white")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def create_comparison_charts(summary: pd.DataFrame, best: pd.DataFrame, charts: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    grouped = summary.groupby(["family", "strategy"], as_index=False).agg(
        mean_vehicles=("vehicle_count", "mean"),
        mean_distance=("total_distance", "mean"),
        mean_runtime=("runtime_seconds", "mean"),
        feasible_rate=("feasible", "mean"),
    )
    x = np.arange(len(FAMILIES))
    width = 0.19
    for metric, ylabel, filename, title in [
        ("mean_vehicles", "Mean vehicle count", "01_mean_vehicles_by_strategy", "BIH vehicle count by family and insertion strategy"),
        ("mean_distance", "Mean total distance", "02_mean_distance_by_strategy", "BIH total distance by family and insertion strategy"),
        ("mean_runtime", "Mean runtime (seconds)", "03_mean_runtime_by_strategy", "BIH runtime by family and insertion strategy"),
    ]:
        figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
        for idx, strategy in enumerate(STRATEGIES):
            values = grouped[grouped["strategy"] == strategy].set_index("family")[metric].reindex(FAMILIES)
            axis.bar(x + (idx - 1.5) * width, values, width, label=STRATEGY_LABELS[strategy], color=COLORS[strategy])
        axis.set_xticks(x, FAMILIES)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontweight="bold")
        axis.legend(ncol=2)
        figure.savefig(charts / f"{filename}.png", dpi=250, bbox_inches="tight", facecolor="white")
        figure.savefig(charts / f"{filename}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    counts = best["strategy"].value_counts().reindex(STRATEGIES, fill_value=0)
    bars = axis.bar([STRATEGY_LABELS[s] for s in STRATEGIES], counts, color=[COLORS[s] for s in STRATEGIES])
    axis.bar_label(bars, padding=3)
    axis.set_title("Number of instances won by each BIH strategy", fontweight="bold")
    axis.set_ylabel("Instances selected as lexicographic best")
    axis.tick_params(axis="x", rotation=12)
    figure.savefig(charts / "04_best_strategy_frequency.png", dpi=250, bbox_inches="tight", facecolor="white")
    figure.savefig(charts / "04_best_strategy_frequency.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    values = [summary.loc[summary["strategy"] == strategy, "vehicle_excess_over_capacity_lb"] for strategy in STRATEGIES]
    boxes = axis.boxplot(values, tick_labels=[STRATEGY_LABELS[s] for s in STRATEGIES], patch_artist=True, showfliers=False)
    for patch, strategy in zip(boxes["boxes"], STRATEGIES):
        patch.set_facecolor(COLORS[strategy])
    axis.set_title("Vehicle excess above the aggregate-demand lower bound", fontweight="bold")
    axis.set_ylabel("Vehicles above capacity lower bound")
    axis.tick_params(axis="x", rotation=12)
    figure.savefig(charts / "05_vehicle_excess_over_lower_bound.png", dpi=250, bbox_inches="tight", facecolor="white")
    figure.savefig(charts / "05_vehicle_excess_over_lower_bound.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)

    pivot = summary.pivot(index="instance", columns="strategy", values="vehicle_count").reindex(columns=STRATEGIES)
    figure, axis = plt.subplots(figsize=(10, 14), constrained_layout=True)
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu")
    axis.set_xticks(range(len(STRATEGIES)), [STRATEGY_LABELS[s] for s in STRATEGIES], rotation=20, ha="right")
    axis.set_yticks(range(len(pivot)), pivot.index, fontsize=6)
    axis.set_title("Vehicle-count heatmap for all 60 instances", fontweight="bold")
    figure.colorbar(image, ax=axis, label="Vehicles")
    figure.savefig(charts / "06_vehicle_count_heatmap_all_instances.png", dpi=250, bbox_inches="tight", facecolor="white")
    figure.savefig(charts / "06_vehicle_count_heatmap_all_instances.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def export_workbook(summary: pd.DataFrame, best: pd.DataFrame, family: pd.DataFrame, routes: pd.DataFrame, output: Path, logger: logging.Logger) -> None:
    workbook_path = output / "E2_BIH_Publication_Ready.xlsx"
    try:
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            family.to_excel(writer, sheet_name="Family_Comparison", index=False)
            best.to_excel(writer, sheet_name="Best_Per_Instance", index=False)
            summary.to_excel(writer, sheet_name="All_Strategies", index=False)
            routes.to_excel(writer, sheet_name="Route_Summary", index=False)
            openpyxl = __import__("openpyxl")
            for sheet in writer.sheets.values():
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions
                for cell in sheet[1]:
                    cell.font = openpyxl.styles.Font(bold=True, color="FFFFFF")
                    cell.fill = openpyxl.styles.PatternFill("solid", fgColor="1F4E78")
                    sheet.column_dimensions[cell.column_letter].width = min(30, max(12, len(str(cell.value)) + 2))
        logger.info("Saved workbook: %s", workbook_path)
    except ImportError:
        logger.warning("openpyxl unavailable: CSV outputs were saved, but the Excel workbook was skipped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2 multi-strategy BIH baseline for Homberger1000")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--e1", type=Path, default=DEFAULT_E1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--limit-instances", type=int, default=0, help="Testing only; 0 means all instances")
    parser.add_argument("--all-route-maps", action="store_true", help="Also draw a route map for all 60 best solutions")
    parser.add_argument("--allow-hash-mismatch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    folders = setup_output(args.output)
    logger = setup_logger(args.output)
    logger.info("Starting E2: multi-strategy BIH baseline")
    logger.info("Dataset: %s", args.dataset)
    logger.info("E1 ZIP : %s", args.e1)
    logger.info("Output : %s", args.output)

    try:
        e1_manifest = read_e1_manifest(args.e1)
        lineage = verify_lineage(args.dataset, e1_manifest, args.allow_hash_mismatch)
        logger.info("E1 lineage verified: SHA-256 match = %s", lineage["hash_match"])
        instances = load_instances(args.dataset, set(args.families), args.limit_instances)
        logger.info("Loaded %d instance(s)", len(instances))

        summary_rows: list[dict] = []
        route_rows_all: list[dict] = []
        stop_rows_all: list[dict] = []
        solution_cache: dict[tuple[str, str], list[list[int]]] = {}
        instance_cache = {instance.name: instance for instance in instances}

        for instance_index, instance in enumerate(instances, start=1):
            for strategy_index, strategy in enumerate(STRATEGIES, start=1):
                routes, metrics = construct_bih(instance, strategy)
                solution_cache[(instance.name, strategy)] = routes
                summary_rows.append({
                    "instance": instance.name,
                    "family": instance.family,
                    "strategy": strategy,
                    **metrics,
                })
                route_rows, stop_rows = solution_rows(instance, strategy, routes)
                route_rows_all.extend(route_rows)
                stop_rows_all.extend(stop_rows)
                logger.info(
                    "[%02d/%02d | %d/4] %-12s %-22s | veh=%3d | dist=%10.2f | time=%7.2fs | %s",
                    instance_index,
                    len(instances),
                    strategy_index,
                    instance.name,
                    strategy,
                    metrics["vehicle_count"],
                    metrics["total_distance"],
                    metrics["runtime_seconds"],
                    "FEASIBLE" if metrics["feasible"] else "INVALID",
                )

        summary = pd.DataFrame(summary_rows)
        routes_df = pd.DataFrame(route_rows_all)
        stops_df = pd.DataFrame(stop_rows_all)
        if not bool(summary["feasible"].all()):
            invalid = summary.loc[~summary["feasible"], ["instance", "strategy"]]
            raise ValueError(f"Invalid BIH solutions detected:\n{invalid.to_string(index=False)}")
        best = choose_best(summary)
        family = summary.groupby(["family", "strategy"], as_index=False).agg(
            instances=("instance", "count"),
            feasible_rate=("feasible", "mean"),
            mean_vehicles=("vehicle_count", "mean"),
            best_vehicles=("vehicle_count", "min"),
            mean_distance=("total_distance", "mean"),
            best_distance=("total_distance", "min"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_vehicle_excess_lb=("vehicle_excess_over_capacity_lb", "mean"),
        )

        summary.to_csv(folders["tables"] / "E2_all_strategy_summary.csv", index=False, encoding="utf-8-sig")
        best.to_csv(folders["tables"] / "E2_best_solution_per_instance.csv", index=False, encoding="utf-8-sig")
        family.to_csv(folders["tables"] / "E2_family_comparison.csv", index=False, encoding="utf-8-sig")
        routes_df.to_csv(folders["tables"] / "E2_route_summary.csv", index=False, encoding="utf-8-sig")
        stops_df.to_csv(folders["tables"] / "E2_route_stops_and_schedule.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
        export_workbook(summary, best, family, routes_df, args.output, logger)
        create_comparison_charts(summary, best, folders["charts"])

        representative_names = {
            family_name: sorted([instance.name for instance in instances if instance.family == family_name], key=natural_key)[0]
            for family_name in FAMILIES
            if any(instance.family == family_name for instance in instances)
        }
        for _, row in best.iterrows():
            instance = instance_cache[row["instance"]]
            strategy = row["strategy"]
            routes = solution_cache[(instance.name, strategy)]
            save_solution_json(instance, strategy, routes, row.to_dict(), folders["solutions"])
            if representative_names.get(instance.family) == instance.name or args.all_route_maps:
                plot_route_map(instance, routes, strategy, folders["routes"] / f"{instance.name}_{strategy}")

        elapsed = time.perf_counter() - started
        manifest = {
            "experiment": "E2 - Multi-strategy BIH baseline",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": str(args.dataset),
            "e1_zip": str(args.e1),
            "dataset_sha256": lineage["dataset_sha256"],
            "e1_dataset_sha256": lineage["e1_dataset_sha256"],
            "sha256_match": lineage["hash_match"],
            "instances": len(instances),
            "strategies": list(STRATEGIES),
            "runs": len(summary),
            "feasible_runs": int(summary["feasible"].sum()),
            "elapsed_seconds": elapsed,
            "objective": "Lexicographic: minimize vehicle count, then total Euclidean distance",
            "distance_convention": "Double-precision Euclidean benchmark units",
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__,
            },
        }
        (args.output / "E2_reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("=" * 78)
        logger.info("E2 COMPLETE | feasible runs: %d/%d", int(summary["feasible"].sum()), len(summary))
        logger.info("Elapsed time: %.1f seconds (%.2f minutes)", elapsed, elapsed / 60)
        logger.info("Results: %s", args.output)
        logger.info("=" * 78)
        return 0
    except Exception:
        logger.exception("E2 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
