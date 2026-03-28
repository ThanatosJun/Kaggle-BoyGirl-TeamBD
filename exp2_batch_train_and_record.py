import argparse
import csv
import json
import re
import subprocess
import sys
from shutil import copy2
from datetime import datetime
from pathlib import Path

import yaml


def parse_exp_id(exp_folder_name: str) -> int:
    """Extract numeric id from folder name like exp_003_xxx."""
    m = re.match(r"^exp_(\d+)_", exp_folder_name)
    if not m:
        return -1
    return int(m.group(1))


def list_experiment_dirs(experiments_dir: Path, exp_name: str):
    pattern = f"exp_*_{exp_name}"
    candidates = [p for p in experiments_dir.glob(pattern) if p.is_dir()]
    return sorted(candidates, key=lambda p: parse_exp_id(p.name))


def read_cv_metrics(cv_results_path: Path):
    with cv_results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "mean_accuracy": data.get("mean_accuracy"),
        "mean_f1": data.get("mean_f1"),
        "mean_precision": data.get("mean_precision"),
        "mean_recall": data.get("mean_recall"),
        "std_accuracy": data.get("std_accuracy"),
        "std_f1": data.get("std_f1"),
        "std_precision": data.get("std_precision"),
        "std_recall": data.get("std_recall"),
    }


def append_record(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "config_file",
        "experiment_name",
        "experiment_folder",
        "model_type",
        "imputation_method",
        "status",
        "return_code",
        "mean_accuracy",
        "mean_f1",
        "mean_precision",
        "mean_recall",
        "std_accuracy",
        "std_f1",
        "std_precision",
        "std_recall",
        "error_message",
    ]

    file_exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def backup_existing_csv(csv_path: Path):
    """Backup existing record file before appending new rows."""
    if not csv_path.exists():
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = csv_path.with_name(f"{csv_path.stem}.bak_{ts}{csv_path.suffix}")
    copy2(csv_path, backup_path)
    return backup_path


def _to_float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def load_latest_success_by_config(csv_path: Path):
    """Load latest successful record per config for before/after comparison."""
    latest = {}
    if not csv_path.exists():
        return latest

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            config_file = row.get("config_file")
            status = row.get("status")
            if not config_file or status != "success":
                continue
            latest[config_file] = row

    return latest


def write_comparison_csv(csv_path: Path, rows: list):
    if not rows:
        return None

    out_path = csv_path.with_name(f"{csv_path.stem}_latest_comparison.csv")
    fieldnames = [
        "config_file",
        "status",
        "previous_experiment_folder",
        "new_experiment_folder",
        "prev_f1",
        "new_f1",
        "delta_f1",
        "prev_accuracy",
        "new_accuracy",
        "delta_accuracy",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return out_path


def summarize_comparison(run_rows: list, previous_map: dict):
    """Build comparison rows and console summary for this run."""
    comparison_rows = []

    improved_f1 = 0
    degraded_f1 = 0
    same_f1 = 0
    new_configs = 0

    for row in run_rows:
        cfg = row["config_file"]
        prev = previous_map.get(cfg)

        new_f1 = _to_float(row.get("mean_f1"))
        new_acc = _to_float(row.get("mean_accuracy"))

        prev_f1 = _to_float(prev.get("mean_f1")) if prev else None
        prev_acc = _to_float(prev.get("mean_accuracy")) if prev else None

        delta_f1 = (new_f1 - prev_f1) if (new_f1 is not None and prev_f1 is not None) else None
        delta_acc = (new_acc - prev_acc) if (new_acc is not None and prev_acc is not None) else None

        if row.get("status") != "success":
            pass
        elif prev_f1 is None:
            new_configs += 1
        elif delta_f1 is not None:
            if delta_f1 > 1e-12:
                improved_f1 += 1
            elif delta_f1 < -1e-12:
                degraded_f1 += 1
            else:
                same_f1 += 1

        comparison_rows.append({
            "config_file": cfg,
            "status": row.get("status"),
            "previous_experiment_folder": prev.get("experiment_folder", "") if prev else "",
            "new_experiment_folder": row.get("experiment_folder", ""),
            "prev_f1": "" if prev_f1 is None else f"{prev_f1:.6f}",
            "new_f1": "" if new_f1 is None else f"{new_f1:.6f}",
            "delta_f1": "" if delta_f1 is None else f"{delta_f1:+.6f}",
            "prev_accuracy": "" if prev_acc is None else f"{prev_acc:.6f}",
            "new_accuracy": "" if new_acc is None else f"{new_acc:.6f}",
            "delta_accuracy": "" if delta_acc is None else f"{delta_acc:+.6f}",
        })

    print("\n=== Comparison vs Previous Run (Exp2) ===")
    print(
        f"Configs: {len(run_rows)} | "
        f"Improved F1: {improved_f1} | Degraded F1: {degraded_f1} | "
        f"Same F1: {same_f1} | New configs: {new_configs}"
    )

    for r in comparison_rows:
        if r["status"] != "success":
            print(f"[failed] {r['config_file']} -> no comparison")
            continue
        if not r["prev_f1"]:
            print(
                f"[new] {r['config_file']} -> "
                f"F1={r['new_f1']}, Acc={r['new_accuracy']}"
            )
            continue
        print(
            f"[cmp] {r['config_file']} -> "
            f"F1 {r['prev_f1']} -> {r['new_f1']} ({r['delta_f1']}), "
            f"Acc {r['prev_accuracy']} -> {r['new_accuracy']} ({r['delta_accuracy']})"
        )

    return comparison_rows


def run_config(config_path: Path, project_root: Path, experiments_dir: Path, main_script: Path):
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["experiment"]["name"]
    model_type = cfg.get("model", {}).get("type")
    imputation_method = cfg.get("preprocessing", {}).get("imputation_method")

    before = set(p.name for p in list_experiment_dirs(experiments_dir, exp_name))

    print(f"\n=== Running {config_path.name} ===")
    cmd = [sys.executable, str(main_script), "--config", str(config_path)]
    result = subprocess.run(cmd, cwd=str(project_root), text=True)

    after_dirs = list_experiment_dirs(experiments_dir, exp_name)
    after = set(p.name for p in after_dirs)
    created = sorted(list(after - before), key=parse_exp_id)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_file": str(config_path.relative_to(project_root)),
        "experiment_name": exp_name,
        "experiment_folder": "",
        "model_type": model_type,
        "imputation_method": imputation_method,
        "status": "success" if result.returncode == 0 else "failed",
        "return_code": result.returncode,
        "mean_accuracy": "",
        "mean_f1": "",
        "mean_precision": "",
        "mean_recall": "",
        "std_accuracy": "",
        "std_f1": "",
        "std_precision": "",
        "std_recall": "",
        "error_message": "",
    }

    target_exp_dir = None
    if created:
        target_exp_dir = experiments_dir / created[-1]
    elif after_dirs:
        # Fallback when no new folder was detected
        target_exp_dir = after_dirs[-1]

    if target_exp_dir is not None:
        row["experiment_folder"] = target_exp_dir.name

    if result.returncode == 0 and target_exp_dir is not None:
        cv_path = target_exp_dir / "cv_results.json"
        if cv_path.exists():
            metrics = read_cv_metrics(cv_path)
            row.update(metrics)
        else:
            row["status"] = "failed"
            row["error_message"] = "cv_results.json not found"
    elif result.returncode != 0:
        row["error_message"] = f"training command failed (code={result.returncode})"

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Batch train Exp2 configs and write summary metrics to experiment_record_exp2.csv"
    )
    parser.add_argument(
        "--config-glob",
        default="configs/exp2_*_method*.yaml",
        help="Glob pattern for config files",
    )
    parser.add_argument(
        "--output-csv",
        default="experiments/experiment_record_exp2.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--main-script",
        default="main_train.py",
        help="Training entry script path",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop batch immediately if one config fails",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    experiments_dir = project_root / "experiments"
    output_csv = project_root / args.output_csv
    main_script = project_root / args.main_script

    previous_map = load_latest_success_by_config(output_csv)
    backup_path = backup_existing_csv(output_csv)
    if backup_path is not None:
        print(f"Backup previous records: {backup_path}")

    config_paths = sorted(project_root.glob(args.config_glob))
    if not config_paths:
        print(f"No config files matched: {args.config_glob}")
        sys.exit(1)

    print(f"Found {len(config_paths)} configs")

    run_rows = []
    for config_path in config_paths:
        row = run_config(config_path, project_root, experiments_dir, main_script)
        append_record(output_csv, row)
        run_rows.append(row)

        print(
            f"[{row['status']}] {row['config_file']} -> "
            f"Acc={row['mean_accuracy']}, F1={row['mean_f1']}, "
            f"Precision={row['mean_precision']}, Recall={row['mean_recall']}"
        )

        if row["status"] != "success" and args.stop_on_error:
            print("Stop on error enabled, aborting batch.")
            sys.exit(1)

    comparison_rows = summarize_comparison(run_rows, previous_map)
    cmp_csv = write_comparison_csv(output_csv, comparison_rows)
    if cmp_csv is not None:
        print(f"Comparison CSV written to: {cmp_csv}")

    print(f"\nDone. Records written to: {output_csv}")


if __name__ == "__main__":
    main()
