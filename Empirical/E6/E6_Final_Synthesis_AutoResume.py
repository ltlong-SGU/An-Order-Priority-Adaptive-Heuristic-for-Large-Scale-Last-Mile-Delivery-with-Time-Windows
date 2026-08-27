# -*- coding: utf-8 -*-
"""E6: final cross-stage audit and publication-ready synthesis.

This program reads E1.zip through E5.zip plus the original Homberger-1000
dataset.  It does not rerun an optimizer.  Instead, it verifies provenance,
checks completeness and consistency, recomputes the principal paired results,
adds effect-size and seed-stability analyses, and exports publication-ready
CSV, Excel, PNG and PDF artifacts.

Normal run (uses the Windows paths requested by the study):
    python E6_Final_Synthesis_AutoResume.py

Safe test without shutting down Windows:
    python E6_Final_Synthesis_AutoResume.py --no-shutdown

Rerun from the beginning (normally unnecessary):
    python E6_Final_Synthesis_AutoResume.py --restart --no-shutdown
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import platform
import subprocess
import sys
import time
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
    print("Install with: python -m pip install numpy pandas matplotlib openpyxl scipy")
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
DEFAULTS = {stage: Path(ROOT + rf"\{stage}.zip") for stage in ("E1", "E2", "E3", "E4", "E5")}
DEFAULT_DATASET = Path(ROOT + r"\homberger_1000_customer_instances.zip")
DEFAULT_OUTPUT = Path(ROOT + r"\E6")
FAMILIES = ("C1", "C2", "R1", "R2", "RC1", "RC2")
EPS = 1e-8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(name: str) -> str:
    return name.replace("\\", "/").lower()


def zip_member(zf: zipfile.ZipFile, suffix: str) -> str:
    wanted = normalized(suffix)
    matches = [name for name in zf.namelist() if normalized(name).endswith(wanted)]
    if not matches:
        raise FileNotFoundError(f"Missing {suffix!r} in {zf.filename}")
    return min(matches, key=len)


def read_json(zf: zipfile.ZipFile, suffix: str) -> dict:
    return json.loads(zf.read(zip_member(zf, suffix)).decode("utf-8-sig"))


def read_csv(zf: zipfile.ZipFile, suffix: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(zf.read(zip_member(zf, suffix))))


def setup_folders(root: Path) -> dict[str, Path]:
    folders = {"root": root, "tables": root / "tables", "charts": root / "charts",
               "checkpoints": root / "checkpoints"}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def setup_logger(root: Path) -> logging.Logger:
    logger = logging.getLogger("E6")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.FileHandler(root / "E6_execution.log", mode="a", encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def final_outputs_exist(root: Path) -> bool:
    required = [
        root / "E6_Final_Synthesis_Publication_Ready.xlsx",
        root / "E6_reproducibility_manifest.json",
        root / "tables" / "E6_final_instance_comparison.csv",
        root / "tables" / "E6_effect_size_and_tests.csv",
        root / "tables" / "E6_route_validation.csv",
        root / "charts" / "01_final_fleet_gap.png",
        root / "charts" / "02_matched_seed_outcomes.png",
        root / "charts" / "03_seed_stability.png",
    ]
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E6 final audit and synthesis for Homberger-1000")
    for stage in ("e1", "e2", "e3", "e4", "e5"):
        parser.add_argument(f"--{stage}", type=Path, default=DEFAULTS[stage.upper()])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--restart", action="store_true", help="Ignore the saved E6 checkpoint")
    parser.add_argument("--no-shutdown", action="store_true", help="Do not shut down Windows after success")
    parser.add_argument("--shutdown-delay", type=int, default=60, help="Shutdown delay in seconds (default: 60)")
    return parser.parse_args()


def validate_zip(path: Path) -> tuple[int, int]:
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad:
            raise zipfile.BadZipFile(f"Corrupt member {bad!r} in {path}")
        files = [item for item in zf.infolist() if not item.is_dir()]
        return len(files), sum(item.file_size for item in files)


def verify_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, dict, str]:
    paths = {stage: getattr(args, stage.lower()) for stage in ("E1", "E2", "E3", "E4", "E5")}
    paths["DATASET"] = args.dataset
    rows, manifests = [], {}
    for stage, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        count, unpacked = validate_zip(path)
        rows.append({"input": stage, "path": str(path), "sha256": sha256_file(path),
                     "archive_files": count, "uncompressed_bytes": unpacked, "zip_valid": True})
        if stage != "DATASET":
            with zipfile.ZipFile(path) as zf:
                manifests[stage] = read_json(zf, f"{stage}_reproducibility_manifest.json")

    dataset_hash = sha256_file(args.dataset)
    for stage, manifest in manifests.items():
        recorded = manifest.get("dataset_sha256")
        if recorded != dataset_hash:
            raise ValueError(f"{stage} dataset SHA-256 mismatch: recorded={recorded}; actual={dataset_hash}")
    return pd.DataFrame(rows), manifests, dataset_hash


def require_columns(frame: pd.DataFrame, name: str, columns: set[str]) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def lexicographic_outcome(av: float, ad: float, bv: float, bd: float) -> str:
    if av < bv:
        return "win"
    if av > bv:
        return "loss"
    if ad < bd - EPS:
        return "win"
    if ad > bd + EPS:
        return "loss"
    return "tie"


def rank_biserial(first: np.ndarray, second: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation; positive favors smaller first values."""
    difference = np.asarray(second, float) - np.asarray(first, float)
    nonzero = difference[np.abs(difference) > EPS]
    if not len(nonzero):
        return 0.0
    order = np.argsort(np.abs(nonzero))
    ranks = np.empty(len(nonzero), float)
    ranks[order] = np.arange(1, len(nonzero) + 1, dtype=float)
    # Average ranks for exact ties.
    magnitudes = np.abs(nonzero)
    for value in np.unique(magnitudes):
        mask = np.isclose(magnitudes, value, atol=EPS, rtol=0)
        ranks[mask] = ranks[mask].mean()
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative))


def paired_pvalue(first: np.ndarray, second: np.ndarray) -> tuple[str, float]:
    delta = np.asarray(first, float) - np.asarray(second, float)
    if not np.any(np.abs(delta) > EPS):
        return ("Wilcoxon signed-rank" if SCIPY_AVAILABLE else "Exact sign test", 1.0)
    if SCIPY_AVAILABLE:
        return "Wilcoxon signed-rank", float(wilcoxon(first, second, zero_method="wilcox").pvalue)
    wins = int(np.sum(delta < -EPS)); losses = int(np.sum(delta > EPS)); n = wins + losses
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2 ** n)
    return "Exact sign test", min(1.0, 2.0 * tail)


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    result = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        result[int(index)] = running
    return result


def build_effect_sizes(paired: pd.DataFrame) -> pd.DataFrame:
    independent = paired.groupby(["instance", "family"], as_index=False).agg(
        alns_vehicles=("alns_vehicles", "mean"), alns_distance=("alns_distance", "mean"),
        da_adr_vehicles=("da_adr_vehicles", "mean"), da_adr_distance=("da_adr_distance", "mean"),
        seeds=("seed", "nunique"))
    rows = []
    for family in ("ALL", *FAMILIES):
        part = independent if family == "ALL" else independent[independent.family == family]
        for metric in ("vehicles", "distance"):
            first = part[f"da_adr_{metric}"].to_numpy(float)
            second = part[f"alns_{metric}"].to_numpy(float)
            test, pvalue = paired_pvalue(first, second)
            outcomes = [lexicographic_outcome(a, 0, b, 0) for a, b in zip(first, second)]
            rows.append({"family": family, "metric": metric, "independent_instances": len(part),
                         "matched_seed_runs": int(part.seeds.sum()), "test": test, "p_value": pvalue,
                         "mean_da_adr": float(first.mean()), "mean_alns": float(second.mean()),
                         "median_da_minus_alns": float(np.median(first - second)),
                         "rank_biserial_effect_positive_favors_da_adr": rank_biserial(first, second),
                         "da_adr_wins": outcomes.count("win"), "ties": outcomes.count("tie"),
                         "da_adr_losses": outcomes.count("loss")})
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_value"] = holm_adjust(result.p_value.tolist())
    result["significant_at_0_05"] = result.holm_adjusted_p_value < 0.05
    return result


def build_seed_stability(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (instance, family), part in paired.groupby(["instance", "family"]):
        for algorithm in ("alns", "da_adr"):
            distance = part[f"{algorithm}_distance"].to_numpy(float)
            vehicles = part[f"{algorithm}_vehicles"].to_numpy(float)
            rows.append({"instance": instance, "family": family, "algorithm": algorithm.upper(),
                         "seeds": part.seed.nunique(), "mean_vehicles": vehicles.mean(),
                         "sd_vehicles": vehicles.std(ddof=1) if len(vehicles) > 1 else 0.0,
                         "mean_distance": distance.mean(),
                         "sd_distance": distance.std(ddof=1) if len(distance) > 1 else 0.0,
                         "cv_distance_percent": (100.0 * distance.std(ddof=1) / distance.mean()
                                                 if len(distance) > 1 and distance.mean() else 0.0),
                         "best_distance": distance.min(), "worst_distance": distance.max()})
    return pd.DataFrame(rows)


def save_plot(fig, folder: Path, name: str) -> None:
    fig.savefig(folder / f"{name}.png", dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_charts(comparison: pd.DataFrame, paired: pd.DataFrame, stability: pd.DataFrame,
                  charts: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False})
    order = list(FAMILIES); x = np.arange(len(order))
    group = comparison.groupby("family").mean(numeric_only=True).reindex(order)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - .18, group.alns_vehicle_gap_vs_bks, .36, label="Standard ALNS", color="#e59739")
    ax.bar(x + .18, group.da_adr_vehicle_gap_vs_bks, .36, label="DA-ADR", color="#2c73a8")
    ax.set_xticks(x, order); ax.set_ylabel("Mean additional vehicles over BKS")
    ax.set_title("Final fleet-size gap to SINTEF BKS"); ax.legend()
    save_plot(fig, charts, "01_final_fleet_gap")

    counts = paired.groupby(["family", "da_adr_vs_alns_lexicographic"]).size().unstack(fill_value=0).reindex(order)
    fig, ax = plt.subplots(figsize=(9.2, 4.8)); bottom = np.zeros(len(order))
    for outcome, color in (("win", "#2c73a8"), ("tie", "#a6adb4"), ("loss", "#d86b5c")):
        values = counts[outcome].to_numpy() if outcome in counts else np.zeros(len(order))
        ax.bar(x, values, bottom=bottom, label=outcome.title(), color=color); bottom += values
    ax.set_xticks(x, order); ax.set_ylabel("Matched seed runs")
    ax.set_title("DA-ADR versus standard ALNS across matched seeds"); ax.legend(ncol=3)
    save_plot(fig, charts, "02_matched_seed_outcomes")

    stab = stability.groupby(["family", "algorithm"], as_index=False).cv_distance_percent.mean()
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for offset, algorithm, color in ((-.18, "ALNS", "#e59739"), (.18, "DA_ADR", "#2c73a8")):
        values = stab[stab.algorithm == algorithm].set_index("family").reindex(order).cv_distance_percent
        ax.bar(x + offset, values, .36, label=algorithm.replace("_", "-"), color=color)
    ax.set_xticks(x, order); ax.set_ylabel("Mean distance CV across seeds (%)")
    ax.set_title("Seed-to-seed stability"); ax.legend()
    save_plot(fig, charts, "03_seed_stability")


def schedule_shutdown(logger: logging.Logger, delay: int) -> None:
    command = ["shutdown", "/s", "/t", str(max(0, delay)), "/c",
               "E6 completed successfully. All experiment results have been saved."]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        logger.info("Windows shutdown scheduled in %d seconds. Cancel with: shutdown /a", delay)
    elif result.returncode == 1190:
        logger.info("Windows already has a scheduled shutdown (code 1190); no duplicate command was added")
    else:
        detail = (result.stderr or result.stdout or "no diagnostic text").strip()
        logger.warning("Results are safe, but shutdown was not scheduled (code %d): %s",
                       result.returncode, detail)


def main() -> int:
    args = parse_args()
    folders = setup_folders(args.output)
    logger = setup_logger(args.output)
    started = time.perf_counter()
    checkpoint_path = folders["checkpoints"] / "E6_progress.json"
    logger.info("Starting E6 final cross-stage audit and synthesis")
    for name in ("e1", "e2", "e3", "e4", "e5", "dataset", "output"):
        logger.info("%s = %s", name.upper(), getattr(args, name))
    try:
        if args.restart and checkpoint_path.exists():
            checkpoint_path.unlink()
        inventory, manifests, dataset_hash = verify_inputs(args)
        input_signature = dict(zip(inventory.input, inventory.sha256))
        saved = load_checkpoint(checkpoint_path)
        if (not args.restart and saved.get("step") == "complete" and
                saved.get("input_signature") == input_signature and final_outputs_exist(args.output)):
            logger.info("RESUME: E6 was already completed for these exact input files; all existing results were reused")
            logger.info("No optimizer or analysis was run again | Results = %s", args.output)
            if platform.system() == "Windows" and not args.no_shutdown:
                schedule_shutdown(logger, args.shutdown_delay)
            else:
                logger.info("Automatic shutdown skipped")
            return 0
        atomic_json(checkpoint_path, {"step": "input_validation_complete", "created_at": datetime.now().isoformat()})
        logger.info("All six archives passed ZIP and SHA-256 lineage checks")

        with zipfile.ZipFile(args.e5) as z5:
            paired = read_csv(z5, "E5_paired_seed_comparison.csv")
            comparison = read_csv(z5, "E5_instance_benchmark_comparison.csv")
            family_summary = read_csv(z5, "E5_family_benchmark_summary.csv")
            validations = read_csv(z5, "E5_independent_route_validation.csv")
            e5_stats = read_csv(z5, "E5_statistical_tests.csv")
            alns_runs = read_csv(z5, "E5_standard_ALNS_all_runs.csv")
            e5_manifest = read_json(z5, "E5_reproducibility_manifest.json")

        require_columns(paired, "E5 paired results", {"instance", "family", "seed", "alns_vehicles",
                        "alns_distance", "da_adr_vehicles", "da_adr_distance",
                        "da_adr_vs_alns_lexicographic"})
        require_columns(comparison, "E5 instance comparison", {"instance", "family", "bks_vehicles",
                        "alns_vehicle_gap_vs_bks", "da_adr_vehicle_gap_vs_bks"})
        require_columns(validations, "E5 route validation", {"instance", "algorithm", "feasible",
                        "missing_customers", "duplicate_customers", "capacity_violations",
                        "time_window_violations", "depot_violations"})

        expected_instances = int(e5_manifest.get("instances", 60))
        expected_runs = int(e5_manifest.get("runs_per_instance", 5))
        if comparison.instance.nunique() != expected_instances:
            raise ValueError("E5 instance comparison is incomplete")
        counts = paired.groupby("instance").seed.nunique()
        if len(counts) != expected_instances or not (counts == expected_runs).all():
            raise ValueError("E5 does not contain the expected matched seeds for every instance")
        if paired.duplicated(["instance", "seed"]).any() or alns_runs.duplicated(["instance", "seed"]).any():
            raise ValueError("Duplicate instance/seed rows detected")
        violation_columns = ["missing_customers", "duplicate_customers", "capacity_violations",
                             "time_window_violations", "depot_violations"]
        if not validations.feasible.astype(bool).all() or validations[violation_columns].to_numpy().sum() != 0:
            raise ValueError("At least one independently validated route is infeasible")

        recomputed = [lexicographic_outcome(row.da_adr_vehicles, row.da_adr_distance,
                                             row.alns_vehicles, row.alns_distance)
                      for row in paired.itertuples()]
        if recomputed != paired.da_adr_vs_alns_lexicographic.astype(str).tolist():
            raise ValueError("Stored paired lexicographic outcomes do not match recomputation")

        completeness = paired.groupby("family", as_index=False).agg(
            instances=("instance", "nunique"), matched_runs=("seed", "count"),
            unique_seeds=("seed", "nunique"))
        completeness["expected_instances"] = expected_instances // len(FAMILIES)
        completeness["expected_runs"] = completeness.expected_instances * expected_runs
        completeness["complete"] = ((completeness.instances == completeness.expected_instances) &
                                     (completeness.matched_runs == completeness.expected_runs))
        stability = build_seed_stability(paired)
        effects = build_effect_sizes(paired)
        claims = pd.DataFrame([
            {"claim": "All experimental stages use the same dataset bytes", "supported": True,
             "evidence": f"SHA-256 {dataset_hash}"},
            {"claim": "E5 covers all instances and matched seeds", "supported": bool(completeness.complete.all()),
             "evidence": f"{expected_instances} instances x {expected_runs} seeds"},
            {"claim": "All independently checked routes are feasible", "supported": True,
             "evidence": f"{len(validations)} route validations; zero recorded violations"},
            {"claim": "DA-ADR is statistically superior after Holm correction", "supported": bool(effects.significant_at_0_05.any()),
             "evidence": "Use the E6 effect-size table; avoid a superiority claim when False"},
            {"claim": "BKS distances are directly comparable regardless of fleet size", "supported": False,
             "evidence": "Distance gaps have hierarchical meaning only at equal fleet size"},
        ])
        atomic_json(checkpoint_path, {"step": "analysis_complete", "created_at": datetime.now().isoformat()})

        tables = {
            "E6_input_inventory.csv": inventory,
            "E6_run_completeness.csv": completeness,
            "E6_final_instance_comparison.csv": comparison,
            "E6_family_summary.csv": family_summary,
            "E6_paired_seed_results.csv": paired,
            "E6_effect_size_and_tests.csv": effects,
            "E6_E5_statistical_tests_preserved.csv": e5_stats,
            "E6_seed_stability.csv": stability,
            "E6_route_validation.csv": validations,
            "E6_publication_claim_audit.csv": claims,
        }
        for filename, frame in tables.items():
            frame.to_csv(folders["tables"] / filename, index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(args.output / "E6_Final_Synthesis_Publication_Ready.xlsx", engine="openpyxl") as writer:
            for title, frame in (("Input inventory", inventory), ("Completeness", completeness),
                                 ("Instance comparison", comparison), ("Family summary", family_summary),
                                 ("Paired seeds", paired), ("Effects and tests", effects),
                                 ("E5 tests preserved", e5_stats), ("Seed stability", stability),
                                 ("Route validation", validations), ("Claim audit", claims)):
                frame.to_excel(writer, sheet_name=title, index=False)
        create_charts(comparison, paired, stability, folders["charts"])

        elapsed = time.perf_counter() - started
        manifest = {
            "experiment": "E6 - final cross-stage integrity audit and publication synthesis",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "inputs": {stage: str(getattr(args, stage.lower())) for stage in ("E1", "E2", "E3", "E4", "E5")}
                      | {"dataset": str(args.dataset)},
            "input_archive_sha256": dict(zip(inventory.input, inventory.sha256)),
            "dataset_sha256": dataset_hash,
            "instances": expected_instances, "runs_per_instance": expected_runs,
            "matched_seed_rows": len(paired), "independent_route_validations": len(validations),
            "all_archives_valid": True, "lineage_consistent": True,
            "all_runs_complete": bool(completeness.complete.all()),
            "all_independent_route_validations_passed": True,
            "statistics": "instance-level paired tests with Holm-Bonferroni correction",
            "effect_size": "matched-pairs rank-biserial correlation",
            "optimizer_rerun": False,
            "elapsed_seconds": elapsed,
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "numpy": np.__version__, "pandas": pd.__version__,
                            "matplotlib": matplotlib.__version__, "scipy_available": SCIPY_AVAILABLE},
        }
        atomic_json(args.output / "E6_reproducibility_manifest.json", manifest)
        atomic_json(checkpoint_path, {"step": "complete", "instances": expected_instances,
                                     "matched_seed_rows": len(paired),
                                     "input_signature": input_signature,
                                     "created_at": datetime.now().isoformat(timespec="seconds")})
        logger.info("E6 COMPLETE | instances=%d | matched runs=%d | validations=%d | all feasible=True",
                    expected_instances, len(paired), len(validations))
        logger.info("Results = %s", args.output)
        if platform.system() == "Windows" and not args.no_shutdown:
            schedule_shutdown(logger, args.shutdown_delay)
        else:
            logger.info("Automatic shutdown skipped")
        return 0
    except Exception:
        logger.exception("E6 failed; automatic shutdown was not requested")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
