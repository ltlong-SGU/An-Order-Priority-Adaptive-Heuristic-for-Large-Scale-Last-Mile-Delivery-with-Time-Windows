#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1: reproducible validation and descriptive analysis of Homberger1000.

Windows quick start:
    python E1_Homberger1000_Validation_Analysis.py

Install dependencies if necessary:
    python -m pip install numpy pandas matplotlib openpyxl

The script reads ZIP archives without extracting them. Distance matrices are
computed one instance at a time, validated, and discarded unless
``--save-matrices`` is explicitly requested.
"""

from __future__ import annotations

import argparse
import gc
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
from typing import Iterable

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


DEFAULT_INPUT = Path(
    r"F:\dulieu\Nghien cứu sinh\Nghiên cứu sinh chính thức"
    r"\Paper_An Order-Priority Adaptive Heuristic for Large-Scale Last-Mile Delivery with Time Windows"
    r"\Empirical\homberger_1000_customer_instances.zip"
)
DEFAULT_OUTPUT = Path(
    r"F:\dulieu\Nghien cứu sinh\Nghiên cứu sinh chính thức"
    r"\Paper_An Order-Priority Adaptive Heuristic for Large-Scale Last-Mile Delivery with Time Windows"
    r"\Empirical\E1"
)

FAMILIES = ("C1", "C2", "R1", "R2", "RC1", "RC2")
EXPECTED_CAPACITY = {"C1": 200, "C2": 700, "R1": 200, "R2": 1000, "RC1": 200, "RC2": 1000}
FAMILY_DESCRIPTION = {
    "C1": "Clustered / type 1",
    "C2": "Clustered / type 2",
    "R1": "Random / type 1",
    "R2": "Random / type 2",
    "RC1": "Mixed / type 1",
    "RC2": "Mixed / type 2",
}
COLORS = {
    "C1": "#2667A5",
    "C2": "#63A7D6",
    "R1": "#D06B36",
    "R2": "#EDAF6C",
    "RC1": "#43856E",
    "RC2": "#8CBF91",
}


@dataclass
class Instance:
    name: str
    family: str
    vehicle_limit: int
    capacity: int
    frame: pd.DataFrame
    sha256: str
    source_name: str


def configure_logging(output: Path) -> logging.Logger:
    logger = logging.getLogger("homberger_e1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output / "E1_execution.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def natural_key(name: str) -> tuple:
    family_match = re.match(r"(RC1|RC2|C1|C2|R1|R2)", Path(name).stem.upper())
    family = family_match.group(1) if family_match else "ZZ"
    family_index = FAMILIES.index(family) if family in FAMILIES else 999
    numbers = tuple(int(part) for part in re.findall(r"\d+", Path(name).stem))
    return (family_index, numbers, Path(name).name)


def read_input_files(source: Path) -> list[tuple[str, bytes]]:
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{source}\n"
            "Check the file name and path, or use --input PATH."
        )
    if source.is_dir():
        items = [(str(path.relative_to(source)), path.read_bytes()) for path in source.rglob("*") if path.suffix.lower() == ".txt"]
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise ValueError(f"Corrupted ZIP member: {bad_member}")
            items = [(info.filename, archive.read(info)) for info in archive.infolist() if not info.is_dir() and info.filename.lower().endswith(".txt")]
    else:
        raise ValueError(f"Input must be a ZIP archive or a folder: {source}")
    if not items:
        raise ValueError("No .TXT instances were found in the input.")
    return sorted(items, key=lambda item: natural_key(item[0]))


def parse_instance(source_name: str, payload: bytes) -> Instance:
    text = payload.decode("utf-8-sig", errors="replace")
    lines = [line.strip() for line in text.splitlines()]
    nonempty = [line for line in lines if line]
    if len(nonempty) < 7:
        raise ValueError(f"Incomplete instance file: {source_name}")

    name = nonempty[0].upper()
    family_match = re.match(r"(RC1|RC2|C1|C2|R1|R2)", name)
    if not family_match:
        raise ValueError(f"Unknown family for instance {name}")
    family = family_match.group(1)

    vehicle_header = next((i for i, line in enumerate(lines) if "NUMBER" in line.upper() and "CAPACITY" in line.upper()), None)
    customer_header = next((i for i, line in enumerate(lines) if "CUST" in line.upper() and "DEMAND" in line.upper()), None)
    if vehicle_header is None or customer_header is None:
        raise ValueError(f"Missing vehicle/customer header in {source_name}")

    vehicle_line = next((line for line in lines[vehicle_header + 1 :] if line), "")
    vehicle_values = re.findall(r"[-+]?\d+(?:\.\d+)?", vehicle_line)
    if len(vehicle_values) < 2:
        raise ValueError(f"Cannot parse vehicle limit/capacity in {source_name}")
    vehicle_limit, capacity = (int(float(vehicle_values[0])), int(float(vehicle_values[1])))

    records: list[list[float]] = []
    for line in lines[customer_header + 1 :]:
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            raise ValueError(f"Invalid customer record in {name}: {line}")
        try:
            records.append([float(value) for value in parts[:7]])
        except ValueError as exc:
            raise ValueError(f"Non-numeric customer record in {name}: {line}") from exc

    columns = ["customer_id", "x", "y", "demand", "ready_time", "due_date", "service_time"]
    frame = pd.DataFrame(records, columns=columns)
    frame["customer_id"] = frame["customer_id"].astype(int)
    frame["window_width"] = frame["due_date"] - frame["ready_time"]
    return Instance(name, family, vehicle_limit, capacity, frame, hashlib.sha256(payload).hexdigest(), source_name)


def distance_matrix(frame: pd.DataFrame) -> np.ndarray:
    xy = frame[["x", "y"]].to_numpy(dtype=np.float64)
    return np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])


def analyze_instance(instance: Instance, matrices_dir: Path | None, triangle_samples: int) -> tuple[dict, list[dict], pd.DataFrame]:
    frame = instance.frame.copy()
    depot_rows = frame[frame["customer_id"] == 0]
    issues: list[dict] = []

    def check(condition: bool, code: str, detail: str, severity: str = "ERROR") -> None:
        if not condition:
            issues.append({"instance": instance.name, "family": instance.family, "severity": severity, "check": code, "detail": detail})

    check(len(depot_rows) == 1, "DEPOT_COUNT", f"Expected one depot; found {len(depot_rows)}")
    check(len(frame) == 1001, "LOCATION_COUNT", f"Expected 1001 locations; found {len(frame)}")
    check(frame["customer_id"].is_unique, "UNIQUE_IDS", "Customer identifiers are not unique")
    check(bool(frame[["x", "y", "demand", "ready_time", "due_date", "service_time"]].notna().all().all()), "MISSING_VALUES", "Missing numeric values")
    check(bool((frame["demand"] >= 0).all()), "NONNEGATIVE_DEMAND", "Negative demand detected")
    check(bool((frame["service_time"] >= 0).all()), "NONNEGATIVE_SERVICE", "Negative service time detected")
    check(bool((frame["ready_time"] <= frame["due_date"]).all()), "VALID_TIME_WINDOWS", "READY TIME exceeds DUE DATE")
    check(instance.vehicle_limit > 0, "VEHICLE_LIMIT", f"Invalid vehicle limit: {instance.vehicle_limit}")
    check(instance.capacity > 0, "VEHICLE_CAPACITY", f"Invalid vehicle capacity: {instance.capacity}")
    check(instance.capacity == EXPECTED_CAPACITY.get(instance.family), "EXPECTED_CAPACITY", f"Observed {instance.capacity}; expected {EXPECTED_CAPACITY.get(instance.family)}", "WARNING")

    customers = frame[frame["customer_id"] != 0].copy()
    check(len(customers) == 1000, "CUSTOMER_COUNT", f"Expected 1000 customers; found {len(customers)}")
    check(bool((customers["demand"] <= instance.capacity).all()), "DEMAND_WITHIN_CAPACITY", "At least one customer exceeds vehicle capacity")

    matrix = distance_matrix(frame)
    diagonal_max = float(np.max(np.abs(np.diag(matrix))))
    symmetry_max = float(np.max(np.abs(matrix - matrix.T)))
    check(bool(np.isfinite(matrix).all()), "FINITE_DISTANCES", "Distance matrix contains NaN or infinity")
    check(bool((matrix >= 0).all()), "NONNEGATIVE_DISTANCES", "Negative distance detected")
    check(diagonal_max <= 1e-9, "ZERO_DIAGONAL", f"Maximum diagonal magnitude: {diagonal_max}")
    check(symmetry_max <= 1e-9, "SYMMETRY", f"Maximum symmetry error: {symmetry_max}")

    rng = np.random.default_rng(20260825)
    triples = rng.integers(0, len(frame), size=(triangle_samples, 3))
    triangle_violation = matrix[triples[:, 0], triples[:, 2]] - matrix[triples[:, 0], triples[:, 1]] - matrix[triples[:, 1], triples[:, 2]]
    triangle_max = max(0.0, float(triangle_violation.max()))
    check(triangle_max <= 1e-8, "TRIANGLE_INEQUALITY", f"Maximum sampled violation: {triangle_max}")

    depot_position = int(np.flatnonzero(frame["customer_id"].to_numpy() == 0)[0]) if len(depot_rows) == 1 else 0
    customer_positions = np.flatnonzero(frame["customer_id"].to_numpy() != 0)
    depot = frame.iloc[depot_position]
    depot_distance = matrix[depot_position, customer_positions]
    earliest_arrival = float(depot["ready_time"]) + depot_distance
    service_start = np.maximum(earliest_arrival, customers["ready_time"].to_numpy())
    return_time = service_start + customers["service_time"].to_numpy() + depot_distance
    individually_feasible = (service_start <= customers["due_date"].to_numpy() + 1e-9) & (return_time <= float(depot["due_date"]) + 1e-9)
    infeasible_count = int((~individually_feasible).sum())
    check(infeasible_count == 0, "SINGLE_CUSTOMER_ROUTE", f"{infeasible_count} customers cannot be served on an isolated depot-customer-depot trip")

    nearest_matrix = matrix.copy()
    np.fill_diagonal(nearest_matrix, np.inf)
    nearest_neighbor = np.min(nearest_matrix[customer_positions], axis=1)
    widths = customers["window_width"].to_numpy(dtype=float)
    width_min, width_max = float(widths.min()), float(widths.max())
    urgency = 1.0 - (widths - width_min) / (width_max - width_min + 1e-12)
    relative_distance = depot_distance / (float(depot_distance.max()) + 1e-12)
    relative_load = customers["demand"].to_numpy(dtype=float) / instance.capacity
    preliminary_difficulty = (urgency + relative_distance + relative_load) / 3.0

    customers["instance"] = instance.name
    customers["family"] = instance.family
    customers["capacity"] = instance.capacity
    customers["distance_to_depot"] = depot_distance
    customers["nearest_neighbor_distance"] = nearest_neighbor
    customers["single_route_feasible"] = individually_feasible
    customers["urgency_normalized"] = urgency
    customers["distance_normalized"] = relative_distance
    customers["load_normalized"] = relative_load
    customers["preliminary_difficulty"] = preliminary_difficulty

    total_demand = float(customers["demand"].sum())
    summary = {
        "instance": instance.name,
        "family": instance.family,
        "distribution": FAMILY_DESCRIPTION[instance.family],
        "source_file": instance.source_name,
        "sha256": instance.sha256,
        "locations": len(frame),
        "customers": len(customers),
        "vehicle_limit": instance.vehicle_limit,
        "capacity": instance.capacity,
        "total_demand": total_demand,
        "capacity_vehicle_lower_bound": math.ceil(total_demand / instance.capacity),
        "mean_demand": float(customers["demand"].mean()),
        "std_demand": float(customers["demand"].std(ddof=0)),
        "min_demand": float(customers["demand"].min()),
        "max_demand": float(customers["demand"].max()),
        "mean_window_width": float(widths.mean()),
        "std_window_width": float(widths.std()),
        "min_window_width": width_min,
        "max_window_width": width_max,
        "mean_service_time": float(customers["service_time"].mean()),
        "depot_ready_time": float(depot["ready_time"]),
        "depot_due_date": float(depot["due_date"]),
        "depot_x": float(depot["x"]),
        "depot_y": float(depot["y"]),
        "mean_depot_distance": float(depot_distance.mean()),
        "max_depot_distance": float(depot_distance.max()),
        "mean_nearest_neighbor_distance": float(nearest_neighbor.mean()),
        "mean_preliminary_difficulty": float(preliminary_difficulty.mean()),
        "individually_infeasible_customers": infeasible_count,
        "matrix_rows": int(matrix.shape[0]),
        "matrix_columns": int(matrix.shape[1]),
        "matrix_diagonal_max_abs": diagonal_max,
        "matrix_symmetry_max_abs": symmetry_max,
        "triangle_samples": triangle_samples,
        "triangle_max_violation": triangle_max,
        "errors": sum(issue["severity"] == "ERROR" for issue in issues),
        "warnings": sum(issue["severity"] == "WARNING" for issue in issues),
        "valid": not any(issue["severity"] == "ERROR" for issue in issues),
    }

    if matrices_dir is not None:
        np.savez_compressed(matrices_dir / f"{instance.name}_distance_matrix.npz", distance=matrix)

    del matrix, nearest_matrix
    gc.collect()
    return summary, issues, customers


def family_axes(title: str, figsize: tuple[int, int] = (16, 10)):
    figure, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    figure.suptitle(title, fontsize=16, fontweight="bold")
    return figure, dict(zip(FAMILIES, axes.flat))


def save_figure(figure, folder: Path, name: str) -> None:
    figure.savefig(folder / f"{name}.png", dpi=250, bbox_inches="tight", facecolor="white")
    figure.savefig(folder / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def representative_customers(customers: pd.DataFrame, family: str) -> pd.DataFrame:
    selected = customers[customers["family"] == family]
    first_instance = sorted(selected["instance"].unique(), key=natural_key)[0]
    return selected[selected["instance"] == first_instance]


def create_charts(instances: pd.DataFrame, customers: pd.DataFrame, charts_dir: Path, logger: logging.Logger) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

    figure, axes = family_axes("Representative delivery-point distribution by Homberger family")
    for family, axis in axes.items():
        sample = representative_customers(customers, family)
        row = instances[instances["instance"] == sample["instance"].iloc[0]].iloc[0]
        axis.scatter(sample["x"], sample["y"], s=9, alpha=0.60, c=COLORS[family], rasterized=True)
        axis.scatter([row["depot_x"]], [row["depot_y"]], marker="*", s=180, c="#C62828", edgecolors="black", label="Depot")
        axis.set_title(f"{family}: {sample['instance'].iloc[0]}")
        axis.set_xlabel("Synthetic X coordinate")
        axis.set_ylabel("Synthetic Y coordinate")
        axis.legend(loc="upper right", fontsize=8)
    save_figure(figure, charts_dir, "01_spatial_distribution_six_families")

    figure, axes = family_axes("Demand distribution across the six instance families")
    for family, axis in axes.items():
        values = customers.loc[customers["family"] == family, "demand"]
        axis.hist(values, bins=20, color=COLORS[family], edgecolor="white")
        axis.set_title(f"{family} | mean = {values.mean():.2f}")
        axis.set_xlabel("Demand (benchmark units)")
        axis.set_ylabel("Number of orders")
    save_figure(figure, charts_dir, "02_order_demand_histograms")

    figure, axes = family_axes("Delivery time-window width by instance family")
    for family, axis in axes.items():
        values = customers.loc[customers["family"] == family, "window_width"]
        axis.hist(values, bins=30, color=COLORS[family], edgecolor="white")
        axis.set_title(f"{family} | mean width = {values.mean():.2f}")
        axis.set_xlabel("Window width (benchmark units)")
        axis.set_ylabel("Number of orders")
    save_figure(figure, charts_dir, "03_time_window_width_histograms")

    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    data = [customers.loc[customers["family"] == family, "demand"].to_numpy() for family in FAMILIES]
    box = axis.boxplot(data, tick_labels=FAMILIES, patch_artist=True, showfliers=False)
    for patch, family in zip(box["boxes"], FAMILIES):
        patch.set_facecolor(COLORS[family])
    axis.set_title("Order demand distribution by family", fontweight="bold")
    axis.set_ylabel("Demand (benchmark units)")
    save_figure(figure, charts_dir, "04_demand_boxplot")

    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    capacities = [EXPECTED_CAPACITY[family] for family in FAMILIES]
    bars = axis.bar(FAMILIES, capacities, color=[COLORS[family] for family in FAMILIES])
    axis.bar_label(bars, padding=4)
    axis.set_title("Vehicle capacity by Homberger family", fontweight="bold")
    axis.set_ylabel("Vehicle capacity (benchmark units)")
    save_figure(figure, charts_dir, "05_vehicle_capacity_by_family")

    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    data = [instances.loc[instances["family"] == family, "capacity_vehicle_lower_bound"].to_numpy() for family in FAMILIES]
    box = axis.boxplot(data, tick_labels=FAMILIES, patch_artist=True)
    for patch, family in zip(box["boxes"], FAMILIES):
        patch.set_facecolor(COLORS[family])
    axis.set_title("Capacity-based lower bound on the number of delivery vehicles", fontweight="bold")
    axis.set_ylabel("Minimum vehicles implied by aggregate demand")
    save_figure(figure, charts_dir, "06_vehicle_lower_bound_boxplot")

    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    family_mean = instances.groupby("family")["mean_nearest_neighbor_distance"].mean().reindex(FAMILIES)
    bars = axis.bar(FAMILIES, family_mean, color=[COLORS[family] for family in FAMILIES])
    axis.bar_label(bars, fmt="%.2f", padding=4)
    axis.set_title("Average nearest-neighbor distance by delivery family", fontweight="bold")
    axis.set_ylabel("Distance (benchmark units)")
    save_figure(figure, charts_dir, "07_nearest_neighbor_distance")

    figure, axes = family_axes("Spatial delivery density: representative instances")
    for family, axis in axes.items():
        sample = representative_customers(customers, family)
        density = axis.hexbin(sample["x"], sample["y"], gridsize=18, cmap="YlOrRd", mincnt=1)
        axis.set_title(f"{family}: {sample['instance'].iloc[0]}")
        axis.set_xlabel("Synthetic X coordinate")
        axis.set_ylabel("Synthetic Y coordinate")
        figure.colorbar(density, ax=axis, label="Orders per cell")
    save_figure(figure, charts_dir, "08_spatial_density_heatmaps")

    figure, axes = family_axes("Preliminary order difficulty from urgency, distance, and load")
    for family, axis in axes.items():
        sample = representative_customers(customers, family)
        row = instances[instances["instance"] == sample["instance"].iloc[0]].iloc[0]
        scatter = axis.scatter(sample["x"], sample["y"], c=sample["preliminary_difficulty"], cmap="viridis", s=13, vmin=0, vmax=1, rasterized=True)
        axis.scatter([row["depot_x"]], [row["depot_y"]], marker="*", s=180, c="#D7263D", edgecolors="black")
        axis.set_title(f"{family}: {sample['instance'].iloc[0]}")
        axis.set_xlabel("Synthetic X coordinate")
        axis.set_ylabel("Synthetic Y coordinate")
        figure.colorbar(scatter, ax=axis, label="Preliminary difficulty")
    save_figure(figure, charts_dir, "09_preliminary_order_difficulty_maps")

    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    correlation_columns = ["demand", "window_width", "distance_to_depot", "nearest_neighbor_distance", "urgency_normalized", "preliminary_difficulty"]
    corr = customers[correlation_columns].corr(numeric_only=True)
    image = axis.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation_columns)), labels=correlation_columns, rotation=35, ha="right")
    axis.set_yticks(range(len(correlation_columns)), labels=correlation_columns)
    for row in range(len(correlation_columns)):
        for column in range(len(correlation_columns)):
            axis.text(column, row, f"{corr.iloc[row, column]:.2f}", ha="center", va="center", color="black", fontsize=8)
    axis.set_title("Correlation among delivery-order characteristics", fontweight="bold")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    save_figure(figure, charts_dir, "10_order_characteristic_correlation")

    figure, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    valid_rates = instances.groupby("family")["valid"].mean().reindex(FAMILIES) * 100
    bars = axis.bar(FAMILIES, valid_rates, color=[COLORS[family] for family in FAMILIES])
    axis.bar_label(bars, fmt="%.1f%%", padding=3)
    axis.set_ylim(0, 110)
    axis.set_title("Instance validation success by family", fontweight="bold")
    axis.set_ylabel("Valid instances (%)")
    save_figure(figure, charts_dir, "11_validation_success_rate")

    figure, axes = family_axes("Urgency versus depot distance, colored by order difficulty")
    for family, axis in axes.items():
        sample = representative_customers(customers, family)
        scatter = axis.scatter(sample["distance_normalized"], sample["urgency_normalized"], c=sample["preliminary_difficulty"], cmap="plasma", s=12, alpha=0.75, rasterized=True)
        axis.set_title(f"{family}: {sample['instance'].iloc[0]}")
        axis.set_xlabel("Normalized depot distance")
        axis.set_ylabel("Normalized time-window urgency")
        figure.colorbar(scatter, ax=axis, label="Preliminary difficulty")
    save_figure(figure, charts_dir, "12_urgency_distance_difficulty_scatter")

    for family in FAMILIES:
        sample = representative_customers(customers, family)
        row = instances[instances["instance"] == sample["instance"].iloc[0]].iloc[0]
        figure, axis = plt.subplots(figsize=(7.5, 7), constrained_layout=True)
        axis.scatter(sample["x"], sample["y"], s=14, color=COLORS[family], alpha=0.72, label="Delivery locations")
        axis.scatter([row["depot_x"]], [row["depot_y"]], s=250, marker="*", color="#C62828", edgecolors="black", label="Depot")
        axis.set_title(f"{family} | {sample['instance'].iloc[0]} | 1,000 delivery locations", fontweight="bold")
        axis.set_xlabel("Synthetic X coordinate")
        axis.set_ylabel("Synthetic Y coordinate")
        axis.legend(loc="upper right")
        save_figure(figure, charts_dir, f"13_{family}_representative_delivery_locations")

    logger.info("Generated 18 chart sets, each saved as PNG and PDF.")


def write_outputs(
    instances: pd.DataFrame,
    customers: pd.DataFrame,
    issues: pd.DataFrame,
    output: Path,
    source: Path,
    started_time: float,
    logger: logging.Logger,
) -> None:
    tables = output / "tables"
    tables.mkdir(exist_ok=True)

    family_summary = instances.groupby("family", sort=False).agg(
        instances=("instance", "count"),
        customers_per_instance=("customers", "first"),
        vehicle_limit=("vehicle_limit", "first"),
        capacity=("capacity", "first"),
        valid_instances=("valid", "sum"),
        mean_total_demand=("total_demand", "mean"),
        mean_vehicle_lower_bound=("capacity_vehicle_lower_bound", "mean"),
        mean_window_width=("mean_window_width", "mean"),
        mean_depot_distance=("mean_depot_distance", "mean"),
        mean_nearest_neighbor_distance=("mean_nearest_neighbor_distance", "mean"),
        individually_infeasible_customers=("individually_infeasible_customers", "sum"),
    ).reindex(FAMILIES).reset_index()
    family_summary.insert(1, "distribution", family_summary["family"].map(FAMILY_DESCRIPTION))

    instance_path = tables / "E1_instance_summary.csv"
    family_path = tables / "E1_family_summary.csv"
    issues_path = tables / "E1_validation_issues.csv"
    customer_path = tables / "E1_all_customer_characteristics.csv"
    instances.to_csv(instance_path, index=False, encoding="utf-8-sig")
    family_summary.to_csv(family_path, index=False, encoding="utf-8-sig")
    issues.to_csv(issues_path, index=False, encoding="utf-8-sig")
    customers.to_csv(customer_path, index=False, encoding="utf-8-sig", float_format="%.8f")

    excel_path = output / "E1_Homberger1000_Publication_Ready.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            family_summary.to_excel(writer, sheet_name="Family_Summary", index=False)
            instances.to_excel(writer, sheet_name="Instance_Summary", index=False)
            issues.to_excel(writer, sheet_name="Validation_Issues", index=False)
            customers.to_excel(writer, sheet_name="All_Customers", index=False)
            for sheet in writer.sheets.values():
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions
                for column in sheet.iter_cols(min_row=1, max_row=1):
                    cell = column[0]
                    cell.font = __import__("openpyxl").styles.Font(bold=True, color="FFFFFF")
                    cell.fill = __import__("openpyxl").styles.PatternFill("solid", fgColor="1F4E78")
                    sheet.column_dimensions[cell.column_letter].width = min(32, max(13, len(str(cell.value)) + 2))
        logger.info("Saved Excel workbook: %s", excel_path)
    except ImportError:
        logger.warning("openpyxl is unavailable. CSV files were saved; install openpyxl to obtain the Excel workbook.")

    archive_hash = sha256_path(source) if source.is_file() else None
    manifest = {
        "experiment": "E1 - Dataset validation and descriptive analysis",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_source": str(source),
        "dataset_sha256": archive_hash,
        "instances": int(len(instances)),
        "customers_total": int(len(customers)),
        "valid_instances": int(instances["valid"].sum()),
        "error_count": int((issues["severity"] == "ERROR").sum()) if not issues.empty else 0,
        "warning_count": int((issues["severity"] == "WARNING").sum()) if not issues.empty else 0,
        "families": family_summary.to_dict(orient="records"),
        "files": instances[["instance", "source_file", "sha256"]].to_dict(orient="records"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "elapsed_seconds": round(time.perf_counter() - started_time, 3),
        "distance_matrix_policy": "Computed sequentially; saved only when --save-matrices is set.",
        "note": "Homberger coordinates and time values are synthetic benchmark units, not real streets, kilometers, or minutes.",
    }
    (output / "E1_reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E1 validation and publication-ready analysis of Homberger1000.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input ZIP archive or folder")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output folder")
    parser.add_argument("--save-matrices", action="store_true", help="Also save compressed 1001x1001 distance matrices")
    parser.add_argument("--triangle-samples", type=int, default=3000, help="Triangle inequality samples per instance")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    started = time.perf_counter()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    charts_dir = output / "charts"
    charts_dir.mkdir(exist_ok=True)
    matrices_dir = output / "distance_matrices" if args.save_matrices else None
    if matrices_dir is not None:
        matrices_dir.mkdir(exist_ok=True)
    logger = configure_logging(output)

    logger.info("Starting E1: Homberger1000 validation and descriptive analysis")
    logger.info("Input : %s", args.input)
    logger.info("Output: %s", output)
    logger.info("Distance matrices: %s", "save compressed matrices" if args.save_matrices else "validate in memory and release")

    try:
        files = read_input_files(args.input)
        logger.info("Found %d TXT instance files", len(files))
        if len(files) != 60:
            logger.warning("Expected 60 files but found %d", len(files))

        summaries: list[dict] = []
        all_issues: list[dict] = []
        customer_frames: list[pd.DataFrame] = []
        for index, (source_name, payload) in enumerate(files, start=1):
            instance = parse_instance(source_name, payload)
            summary, issues, customer_frame = analyze_instance(instance, matrices_dir, args.triangle_samples)
            summaries.append(summary)
            all_issues.extend(issues)
            customer_frames.append(customer_frame)
            logger.info("[%02d/%02d] %-12s | family=%-3s | customers=%4d | Q=%4d | lower_bound=%3d | %s", index, len(files), instance.name, instance.family, summary["customers"], instance.capacity, summary["capacity_vehicle_lower_bound"], "VALID" if summary["valid"] else "INVALID")

        instances = pd.DataFrame(summaries)
        customers = pd.concat(customer_frames, ignore_index=True)
        issues = pd.DataFrame(all_issues, columns=["instance", "family", "severity", "check", "detail"])
        for family in FAMILIES:
            count = int((instances["family"] == family).sum())
            if count != 10:
                logger.warning("Family %s has %d instances; expected 10", family, count)

        logger.info("Producing publication-ready charts...")
        create_charts(instances, customers, charts_dir, logger)
        write_outputs(instances, customers, issues, output, args.input, started, logger)
        elapsed = time.perf_counter() - started

        logger.info("=" * 72)
        logger.info("E1 COMPLETE")
        logger.info("Valid instances: %d/%d", int(instances["valid"].sum()), len(instances))
        logger.info("Total customers: %d", len(customers))
        logger.info("Errors         : %d", int((issues["severity"] == "ERROR").sum()) if not issues.empty else 0)
        logger.info("Warnings       : %d", int((issues["severity"] == "WARNING").sum()) if not issues.empty else 0)
        logger.info("Elapsed time   : %.1f seconds (%.2f minutes)", elapsed, elapsed / 60)
        logger.info("Output folder  : %s", output)
        logger.info("=" * 72)
        return 0 if bool(instances["valid"].all()) else 2
    except Exception:
        logger.exception("E1 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
