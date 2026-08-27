# -*- coding: utf-8 -*-
"""E3: Order-difficulty construction and empirical validation.

Inputs:
  1. Homberger1000 original ZIP
  2. E1 ZIP (dataset validation)
  3. E2 ZIP (best BIH routes)

The proposed score combines time-window urgency, normalized depot distance,
capacity pressure, and route-based slack pressure. Unlike min-max urgency,
the urgency definition is scaled by the depot planning horizon, so instance
families with identical customer-window widths remain well-defined.

The score is validated against an operational proxy: the proportion of route
positions into which a removed order cannot be feasibly reinserted. This gives
reviewers independent evidence that high-scoring orders are genuinely harder
to place in the delivery plan.
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
DEFAULT_E2 = Path(ROOT + r"\E2.zip")
DEFAULT_E1 = Path(ROOT + r"\E1.zip")
DEFAULT_DATASET = Path(ROOT + r"\homberger_1000_customer_instances.zip")
DEFAULT_OUTPUT = Path(ROOT + r"\E3")

FAMILIES = ("C1", "C2", "R1", "R2", "RC1", "RC2")
CONFIGURATIONS = ("P0_uniform", "P1_urgency", "P2_urgency_distance", "P3_static_full", "P4_proposed")
CONFIG_LABELS = {
    "P0_uniform": "P0 Uniform",
    "P1_urgency": "P1 Urgency",
    "P2_urgency_distance": "P2 U+D",
    "P3_static_full": "P3 U+D+C",
    "P4_proposed": "P4 Proposed",
}
CONFIG_COLORS = {
    "P0_uniform": "#A0AEC0",
    "P1_urgency": "#3182CE",
    "P2_urgency_distance": "#38A169",
    "P3_static_full": "#DD6B20",
    "P4_proposed": "#805AD5",
}
WEIGHTS = {"urgency": 0.30, "distance": 0.25, "capacity": 0.20, "slack": 0.25}
EPS = 1e-9


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


@dataclass
class RouteProfile:
    route: list[int]
    earliest: np.ndarray
    latest: np.ndarray
    waiting: np.ndarray
    load: float
    distance: float
    feasible: bool


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
    rank = FAMILIES.index(family) if family in FAMILIES else 999
    return rank, tuple(int(v) for v in re.findall(r"\d+", stem)), stem


def setup_folders(output: Path) -> dict[str, Path]:
    folders = {
        "root": output,
        "tables": output / "tables",
        "charts": output / "charts",
        "maps": output / "difficulty_maps",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def setup_logger(output: Path) -> logging.Logger:
    logger = logging.getLogger("homberger_e3")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(output / "E3_execution.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def read_json_from_zip(path: Path, suffix: str) -> dict:
    if not path.exists() or not zipfile.is_zipfile(path):
        raise FileNotFoundError(f"ZIP not found or invalid: {path}")
    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one {suffix} in {path}; found {len(matches)}")
        return json.loads(archive.read(matches[0]).decode("utf-8-sig"))


def read_e2_solutions(path: Path) -> dict[str, dict]:
    if not path.exists() or not zipfile.is_zipfile(path):
        raise FileNotFoundError(f"E2 ZIP not found or invalid: {path}")
    solutions: dict[str, dict] = {}
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if "/best_solutions_json/" in name.replace("\\", "/") and name.endswith("_best.json")]
        for member in members:
            payload = json.loads(archive.read(member).decode("utf-8-sig"))
            solutions[payload["instance"].upper()] = payload
    if not solutions:
        raise ValueError("No best E2 solutions were found.")
    return solutions


def verify_lineage(dataset: Path, e1: dict, e2: dict, allow_mismatch: bool) -> dict:
    current_hash = sha256_file(dataset)
    e1_hash = str(e1.get("dataset_sha256", ""))
    e2_hash = str(e2.get("dataset_sha256", ""))
    matches = current_hash.lower() == e1_hash.lower() == e2_hash.lower()
    if not matches and not allow_mismatch:
        raise ValueError(
            "Dataset lineage mismatch. E3 requires the identical ZIP used in E1 and E2.\n"
            f"Current: {current_hash}\nE1: {e1_hash}\nE2: {e2_hash}"
        )
    if int(e1.get("valid_instances", 0)) != int(e1.get("instances", -1)):
        raise ValueError("E1 did not validate every instance.")
    if int(e2.get("feasible_runs", 0)) != int(e2.get("runs", -1)):
        raise ValueError("E2 contains infeasible runs.")
    return {"dataset_sha256": current_hash, "e1_sha256": e1_hash, "e2_sha256": e2_hash, "match": matches}


def parse_instance(name: str, payload: bytes) -> Instance:
    lines = [line.strip() for line in payload.decode("utf-8-sig", errors="replace").splitlines()]
    nonempty = [line for line in lines if line]
    instance_name = nonempty[0].upper()
    family_match = re.match(r"(RC1|RC2|C1|C2|R1|R2)", instance_name)
    if not family_match:
        raise ValueError(f"Unknown family in {name}")
    family = family_match.group(1)
    vehicle_header = next(i for i, line in enumerate(lines) if "NUMBER" in line.upper() and "CAPACITY" in line.upper())
    customer_header = next(i for i, line in enumerate(lines) if "CUST" in line.upper() and "DEMAND" in line.upper())
    vehicle_line = next(line for line in lines[vehicle_header + 1 :] if line)
    capacity = float(re.findall(r"[-+]?\d+(?:\.\d+)?", vehicle_line)[1])
    records = []
    for line in lines[customer_header + 1 :]:
        if line:
            parts = line.split()
            if len(parts) >= 7:
                records.append([float(value) for value in parts[:7]])
    data = np.asarray(records, dtype=float)
    if data.shape != (1001, 7):
        raise ValueError(f"{instance_name}: expected (1001, 7), found {data.shape}")
    order = np.argsort(data[:, 0].astype(int))
    data = data[order]
    if not np.array_equal(data[:, 0].astype(int), np.arange(1001)):
        raise ValueError(f"{instance_name}: IDs must be 0..1000")
    x, y = data[:, 1], data[:, 2]
    distance = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    return Instance(instance_name, family, capacity, x, y, data[:, 3], data[:, 4], data[:, 5], data[:, 6], distance)


def load_instances(dataset: Path, selected: set[str], limit: int) -> list[Instance]:
    instances: list[Instance] = []
    with zipfile.ZipFile(dataset) as archive:
        members = sorted([name for name in archive.namelist() if name.lower().endswith(".txt")], key=natural_key)
        for member in members:
            candidate = Path(member).stem.upper()
            family_match = re.match(r"(RC1|RC2|C1|C2|R1|R2)", candidate)
            if family_match and family_match.group(1) in selected:
                instances.append(parse_instance(member, archive.read(member)))
                if limit > 0 and len(instances) >= limit:
                    break
    return instances


def route_profile(instance: Instance, route: list[int]) -> RouteProfile:
    nodes = np.asarray(route, dtype=int)
    n = len(nodes)
    earliest = np.empty(n, dtype=float)
    latest = np.empty(n, dtype=float)
    waiting = np.zeros(n, dtype=float)
    earliest[0] = instance.ready[nodes[0]]
    feasible = route[0] == 0 and route[-1] == 0
    for pos in range(1, n):
        previous, node = nodes[pos - 1], nodes[pos]
        raw = earliest[pos - 1] + instance.service[previous] + instance.distance[previous, node]
        earliest[pos] = max(instance.ready[node], raw)
        waiting[pos] = max(0.0, instance.ready[node] - raw)
        feasible = feasible and earliest[pos] <= instance.due[node] + EPS
    latest[-1] = instance.due[nodes[-1]]
    for pos in range(n - 2, -1, -1):
        node, following = nodes[pos], nodes[pos + 1]
        latest[pos] = min(instance.due[node], latest[pos + 1] - instance.service[node] - instance.distance[node, following])
    load = float(instance.demand[nodes[nodes != 0]].sum())
    feasible = feasible and load <= instance.capacity + EPS
    length = float(instance.distance[nodes[:-1], nodes[1:]].sum())
    return RouteProfile(route, earliest, latest, waiting, load, length, feasible)


def validate_solution(instance: Instance, routes: list[list[int]]) -> None:
    visits = [node for route in routes for node in route if node != 0]
    if len(visits) != 1000 or set(visits) != set(range(1, 1001)) or len(visits) != len(set(visits)):
        raise ValueError(f"{instance.name}: invalid E2 customer coverage")
    if not all(route_profile(instance, route).feasible for route in routes):
        raise ValueError(f"{instance.name}: infeasible E2 route detected")


def order_features(instance: Instance, routes: list[list[int]], strategy: str) -> pd.DataFrame:
    validate_solution(instance, routes)
    horizon = max(EPS, instance.due[0] - instance.ready[0])
    max_depot_distance = max(EPS, float(instance.distance[0, 1:].max()))
    rows: list[dict] = []
    for route_id, route in enumerate(routes, start=1):
        profile = route_profile(instance, route)
        for sequence in range(1, len(route) - 1):
            customer = route[sequence]
            previous, following = route[sequence - 1], route[sequence + 1]
            window_width = instance.due[customer] - instance.ready[customer]
            route_slack = max(0.0, profile.latest[sequence] - profile.earliest[sequence])
            removal_saving = (
                instance.distance[previous, customer]
                + instance.distance[customer, following]
                - instance.distance[previous, following]
            )
            rows.append({
                "instance": instance.name,
                "family": instance.family,
                "e2_strategy": strategy,
                "route_id": route_id,
                "sequence": sequence,
                "customer_id": customer,
                "x": instance.x[customer],
                "y": instance.y[customer],
                "demand": instance.demand[customer],
                "capacity": instance.capacity,
                "ready_time": instance.ready[customer],
                "due_date": instance.due[customer],
                "window_width": window_width,
                "service_time": instance.service[customer],
                "service_start": profile.earliest[sequence],
                "latest_feasible_start": profile.latest[sequence],
                "route_slack": route_slack,
                "waiting_time": profile.waiting[sequence],
                "distance_to_depot": instance.distance[0, customer],
                "removal_saving": removal_saving,
                "U_urgency": float(np.clip(1.0 - window_width / horizon, 0.0, 1.0)),
                "D_distance": instance.distance[0, customer] / max_depot_distance,
                "C_capacity": instance.demand[customer] / instance.capacity,
            })
    frame = pd.DataFrame(rows)
    slack_rank = frame["route_slack"].rank(method="average", pct=True)
    frame["S_slack_pressure"] = 1.0 - slack_rank
    frame["P0_uniform"] = 0.5
    frame["P1_urgency"] = frame["U_urgency"]
    frame["P2_urgency_distance"] = 0.5 * (frame["U_urgency"] + frame["D_distance"])
    frame["P3_static_full"] = (frame["U_urgency"] + frame["D_distance"] + frame["C_capacity"]) / 3.0
    frame["P4_proposed"] = (
        WEIGHTS["urgency"] * frame["U_urgency"]
        + WEIGHTS["distance"] * frame["D_distance"]
        + WEIGHTS["capacity"] * frame["C_capacity"]
        + WEIGHTS["slack"] * frame["S_slack_pressure"]
    )
    return frame


def count_reinsertion_positions(instance: Instance, routes: list[list[int]], customer: int, location: tuple[int, int]) -> dict:
    own_route_id, own_sequence = location
    total_positions = 0
    feasible_positions = 0
    for route_id, original_route in enumerate(routes):
        if route_id == own_route_id:
            route = original_route[:own_sequence] + original_route[own_sequence + 1 :]
        else:
            route = original_route
        profile = route_profile(instance, route)
        positions = len(route) - 1
        total_positions += positions
        if profile.load + instance.demand[customer] > instance.capacity + EPS:
            continue
        nodes = np.asarray(route, dtype=int)
        previous = nodes[:-1]
        following = nodes[1:]
        raw_arrival = profile.earliest[:-1] + instance.service[previous] + instance.distance[previous, customer]
        service_start = np.maximum(instance.ready[customer], raw_arrival)
        following_start = np.maximum(
            instance.ready[following],
            service_start + instance.service[customer] + instance.distance[customer, following],
        )
        feasible = (service_start <= instance.due[customer] + EPS) & (following_start <= profile.latest[1:] + EPS)
        feasible_positions += int(feasible.sum())
    difficulty = 1.0 - feasible_positions / max(1, total_positions)
    return {
        "feasible_reinsertion_positions": feasible_positions,
        "total_candidate_positions": total_positions,
        "reinsertion_difficulty": difficulty,
    }


def validation_sample(instance: Instance, routes: list[list[int]], features: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    size = min(1000, max(20, sample_size))
    sampled_customers = np.unique(np.linspace(1, 1000, size, dtype=int))
    locations = {}
    for route_id, route in enumerate(routes):
        for sequence in range(1, len(route) - 1):
            locations[route[sequence]] = (route_id, sequence)
    indexed = features.set_index("customer_id")
    rows = []
    for customer in sampled_customers:
        counts = count_reinsertion_positions(instance, routes, int(customer), locations[int(customer)])
        row = indexed.loc[int(customer)].to_dict()
        row["customer_id"] = int(customer)
        row.update(counts)
        rows.append(row)
    return pd.DataFrame(rows)


def score_metrics(sample: pd.DataFrame) -> list[dict]:
    rows = []
    n = len(sample)
    k = max(1, math.ceil(0.10 * n))
    actual_top = set(sample.nlargest(k, "reinsertion_difficulty").index)
    for config in CONFIGURATIONS:
        score = sample[config]
        score_rank = score.rank(method="average")
        actual_rank = sample["reinsertion_difficulty"].rank(method="average")
        # Spearman's rho is undefined when either variable is constant.  Record
        # NaN explicitly (rather than emitting NumPy divide-by-zero warnings).
        correlation = (
            float("nan")
            if score_rank.nunique() < 2 or actual_rank.nunique() < 2
            else float(score_rank.corr(actual_rank))
        )
        predicted_top = set(sample.nlargest(k, config).index)
        rows.append({
            "instance": sample["instance"].iloc[0],
            "family": sample["family"].iloc[0],
            "configuration": config,
            "sampled_orders": n,
            "spearman_correlation": correlation,
            "top10_capture_rate": len(actual_top & predicted_top) / k,
            "mean_score": float(score.mean()),
            "std_score": float(score.std(ddof=0)),
            "mean_reinsertion_difficulty": float(sample["reinsertion_difficulty"].mean()),
        })
    return rows


def save_plot(figure, folder: Path, name: str) -> None:
    figure.savefig(folder / f"{name}.png", dpi=250, bbox_inches="tight", facecolor="white")
    figure.savefig(folder / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def create_charts(orders: pd.DataFrame, samples: pd.DataFrame, metrics: pd.DataFrame, family_metrics: pd.DataFrame, charts: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(len(FAMILIES))

    component_means = orders.groupby("family")[["U_urgency", "D_distance", "C_capacity", "S_slack_pressure"]].mean().reindex(FAMILIES)
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    width = 0.2
    for idx, column in enumerate(component_means.columns):
        axis.bar(x + (idx - 1.5) * width, component_means[column], width, label=column)
    axis.set_xticks(x, FAMILIES)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean normalized component")
    axis.set_title("Order-difficulty components by Homberger family", fontweight="bold")
    axis.legend(ncol=2)
    save_plot(figure, charts, "01_difficulty_components_by_family")

    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    values = [orders.loc[orders["family"] == family, "P4_proposed"] for family in FAMILIES]
    boxes = axis.boxplot(values, tick_labels=FAMILIES, patch_artist=True, showfliers=False)
    for patch in boxes["boxes"]:
        patch.set_facecolor("#805AD5")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Proposed difficulty score")
    axis.set_title("Distribution of the proposed order-difficulty score", fontweight="bold")
    save_plot(figure, charts, "02_proposed_difficulty_boxplot")

    for metric_name, ylabel, filename, title in [
        ("spearman_correlation", "Mean Spearman correlation", "03_score_validation_correlation", "Agreement between difficulty scores and reinsertion difficulty"),
        ("top10_capture_rate", "Top-10% capture rate", "04_top10_capture_rate", "Ability to identify the hardest 10% of delivery orders"),
    ]:
        figure, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
        width = 0.16
        for idx, config in enumerate(CONFIGURATIONS):
            data = family_metrics[family_metrics["configuration"] == config].set_index("family")[metric_name].reindex(FAMILIES)
            axis.bar(x + (idx - 2) * width, data, width, color=CONFIG_COLORS[config], label=CONFIG_LABELS[config])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, FAMILIES)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontweight="bold")
        axis.legend(ncol=3)
        save_plot(figure, charts, filename)

    representatives = {family: sorted(orders.loc[orders["family"] == family, "instance"].unique(), key=natural_key)[0] for family in FAMILIES if family in set(orders["family"])}
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    figure.suptitle("Proposed score versus observed reinsertion difficulty", fontsize=16, fontweight="bold")
    for family, axis in zip(representatives, axes.flat):
        instance = representatives[family]
        data = samples[samples["instance"] == instance]
        axis.scatter(data["P4_proposed"], data["reinsertion_difficulty"], s=18, alpha=0.70, color="#805AD5")
        corr = data["P4_proposed"].rank().corr(data["reinsertion_difficulty"].rank())
        axis.set_title(f"{family}: {instance} | rho={corr:.2f}")
        axis.set_xlabel("P4 proposed difficulty")
        axis.set_ylabel("Reinsertion difficulty")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
    save_plot(figure, charts, "05_proposed_vs_reinsertion_difficulty")

    pivot = metrics[metrics["configuration"] == "P4_proposed"].pivot(index="instance", columns="family", values="spearman_correlation")
    ordered_instances = sorted(metrics["instance"].unique(), key=natural_key)
    p4 = metrics[metrics["configuration"] == "P4_proposed"].set_index("instance").reindex(ordered_instances)
    figure, axis = plt.subplots(figsize=(8, 14), constrained_layout=True)
    matrix = p4[["spearman_correlation", "top10_capture_rate"]].to_numpy()
    image = axis.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    axis.set_xticks([0, 1], ["Spearman correlation", "Top-10% capture"])
    axis.set_yticks(range(len(p4)), p4.index, fontsize=6)
    axis.set_title("P4 validation across all instances", fontweight="bold")
    figure.colorbar(image, ax=axis, label="Validation value")
    save_plot(figure, charts, "06_p4_validation_heatmap_all_instances")


def difficulty_map(instance: Instance, routes: list[list[int]], frame: pd.DataFrame, output: Path) -> None:
    threshold = frame["P4_proposed"].quantile(0.90)
    hard = frame[frame["P4_proposed"] >= threshold]
    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    for route in routes:
        nodes = np.asarray(route, dtype=int)
        axis.plot(instance.x[nodes], instance.y[nodes], color="#CBD5E0", linewidth=0.35, alpha=0.30)
    axis.scatter(frame["x"], frame["y"], s=8, color="#A0AEC0", alpha=0.45, label="Other orders")
    scatter = axis.scatter(hard["x"], hard["y"], s=25, c=hard["P4_proposed"], cmap="plasma", edgecolors="black", linewidths=0.2, label="Top 10% difficult")
    axis.scatter(instance.x[0], instance.y[0], marker="*", s=240, color="#D7191C", edgecolors="black", zorder=10, label="Depot")
    axis.set_title(f"{instance.name}: difficult orders on the E2 delivery plan", fontweight="bold")
    axis.set_xlabel("Synthetic X coordinate")
    axis.set_ylabel("Synthetic Y coordinate")
    axis.legend(loc="upper right")
    axis.set_aspect("equal", adjustable="box")
    figure.colorbar(scatter, ax=axis, label="P4 difficulty")
    save_plot(figure, output.parent, output.name)


def export_excel(family: pd.DataFrame, instance_metrics: pd.DataFrame, top: pd.DataFrame, samples: pd.DataFrame, orders: pd.DataFrame, output: Path, logger: logging.Logger) -> None:
    path = output / "E3_Order_Difficulty_Publication_Ready.xlsx"
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            family.to_excel(writer, sheet_name="Family_Validation", index=False)
            instance_metrics.to_excel(writer, sheet_name="Instance_Validation", index=False)
            top.to_excel(writer, sheet_name="Top_Difficult_Orders", index=False)
            samples.to_excel(writer, sheet_name="Reinsertion_Sample", index=False)
            orders.to_excel(writer, sheet_name="All_Order_Scores", index=False)
            openpyxl = __import__("openpyxl")
            for sheet in writer.sheets.values():
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions
                for cell in sheet[1]:
                    cell.font = openpyxl.styles.Font(bold=True, color="FFFFFF")
                    cell.fill = openpyxl.styles.PatternFill("solid", fgColor="1F4E78")
                    sheet.column_dimensions[cell.column_letter].width = min(30, max(12, len(str(cell.value)) + 2))
        logger.info("Saved Excel workbook: %s", path)
    except ImportError:
        logger.warning("openpyxl unavailable; CSV files were saved, but Excel was skipped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E3 order-difficulty validation")
    parser.add_argument("--e2", type=Path, default=DEFAULT_E2)
    parser.add_argument("--e1", type=Path, default=DEFAULT_E1)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=200, help="Orders per instance used for reinsertion validation")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--limit-instances", type=int, default=0, help="Testing only; 0 means all instances")
    parser.add_argument("--allow-hash-mismatch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    folders = setup_folders(args.output)
    logger = setup_logger(args.output)
    logger.info("Starting E3: order-difficulty validation")
    logger.info("E2 ZIP : %s", args.e2)
    logger.info("E1 ZIP : %s", args.e1)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Output : %s", args.output)
    try:
        e1_manifest = read_json_from_zip(args.e1, "E1_reproducibility_manifest.json")
        e2_manifest = read_json_from_zip(args.e2, "E2_reproducibility_manifest.json")
        lineage = verify_lineage(args.dataset, e1_manifest, e2_manifest, args.allow_hash_mismatch)
        solutions = read_e2_solutions(args.e2)
        instances = load_instances(args.dataset, set(args.families), args.limit_instances)
        logger.info("Lineage SHA-256 match: %s", lineage["match"])
        logger.info("Loaded %d instances and %d E2 best solutions", len(instances), len(solutions))

        order_frames: list[pd.DataFrame] = []
        sample_frames: list[pd.DataFrame] = []
        metric_rows: list[dict] = []
        cache: dict[str, tuple[Instance, list[list[int]], pd.DataFrame]] = {}
        for index, instance in enumerate(instances, start=1):
            if instance.name not in solutions:
                raise ValueError(f"Missing E2 best solution: {instance.name}")
            solution = solutions[instance.name]
            routes = [[int(node) for node in route] for route in solution["routes"]]
            strategy = solution["strategy"]
            features = order_features(instance, routes, strategy)
            sample = validation_sample(instance, routes, features, args.sample_size)
            metrics = score_metrics(sample)
            order_frames.append(features)
            sample_frames.append(sample)
            metric_rows.extend(metrics)
            cache[instance.name] = (instance, routes, features)
            p4 = next(row for row in metrics if row["configuration"] == "P4_proposed")
            logger.info(
                "[%02d/%02d] %-12s | strategy=%-22s | routes=%3d | P4 rho=%6.3f | top10=%5.1f%%",
                index,
                len(instances),
                instance.name,
                strategy,
                len(routes),
                p4["spearman_correlation"],
                100 * p4["top10_capture_rate"],
            )

        orders = pd.concat(order_frames, ignore_index=True)
        samples = pd.concat(sample_frames, ignore_index=True)
        metrics = pd.DataFrame(metric_rows)
        family_metrics = metrics.groupby(["family", "configuration"], as_index=False).agg(
            instances=("instance", "count"),
            mean_spearman=("spearman_correlation", "mean"),
            std_spearman=("spearman_correlation", "std"),
            mean_top10_capture=("top10_capture_rate", "mean"),
            mean_score=("mean_score", "mean"),
            mean_reinsertion_difficulty=("mean_reinsertion_difficulty", "mean"),
        )
        family_metrics = family_metrics.rename(columns={"mean_spearman": "spearman_correlation", "mean_top10_capture": "top10_capture_rate"})
        top_orders = orders.sort_values(["instance", "P4_proposed"], ascending=[True, False]).groupby("instance", as_index=False).head(100)

        orders.to_csv(folders["tables"] / "E3_all_order_difficulty_scores.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
        samples.to_csv(folders["tables"] / "E3_reinsertion_validation_sample.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
        metrics.to_csv(folders["tables"] / "E3_instance_validation_metrics.csv", index=False, encoding="utf-8-sig")
        family_metrics.to_csv(folders["tables"] / "E3_family_validation_summary.csv", index=False, encoding="utf-8-sig")
        top_orders.to_csv(folders["tables"] / "E3_top10_percent_difficult_orders.csv", index=False, encoding="utf-8-sig")
        export_excel(family_metrics, metrics, top_orders, samples, orders, args.output, logger)
        create_charts(orders, samples, metrics, family_metrics, folders["charts"])

        representatives = {family: sorted([name for name, value in cache.items() if value[0].family == family], key=natural_key)[0] for family in FAMILIES if any(value[0].family == family for value in cache.values())}
        for family, name in representatives.items():
            instance, routes, features = cache[name]
            difficulty_map(instance, routes, features, folders["maps"] / f"{name}_P4_difficulty_map")

        elapsed = time.perf_counter() - started
        p4_summary = family_metrics[family_metrics["configuration"] == "P4_proposed"]
        manifest = {
            "experiment": "E3 - Order-difficulty validation",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "e2_zip": str(args.e2),
            "e1_zip": str(args.e1),
            "dataset": str(args.dataset),
            **lineage,
            "instances": len(instances),
            "orders": len(orders),
            "sample_size_per_instance": args.sample_size,
            "sampled_orders": len(samples),
            "configurations": list(CONFIGURATIONS),
            "proposed_weights": WEIGHTS,
            "mean_p4_spearman": float(p4_summary["spearman_correlation"].mean()),
            "mean_p4_top10_capture": float(p4_summary["top10_capture_rate"].mean()),
            "elapsed_seconds": elapsed,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__,
            },
        }
        (args.output / "E3_reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("=" * 78)
        logger.info("E3 COMPLETE | instances=%d | orders=%d | validation sample=%d", len(instances), len(orders), len(samples))
        logger.info("Mean P4 Spearman: %.4f", manifest["mean_p4_spearman"])
        logger.info("Mean P4 top-10%% capture: %.2f%%", 100 * manifest["mean_p4_top10_capture"])
        logger.info("Elapsed time: %.1f seconds (%.2f minutes)", elapsed, elapsed / 60)
        logger.info("Results: %s", args.output)
        logger.info("=" * 78)
        return 0
    except Exception:
        logger.exception("E3 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
