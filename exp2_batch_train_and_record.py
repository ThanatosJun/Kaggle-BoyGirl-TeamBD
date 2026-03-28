import argparse
import csv
import json
import re
import subprocess
import sys
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

    config_paths = sorted(project_root.glob(args.config_glob))
    if not config_paths:
        print(f"No config files matched: {args.config_glob}")
        sys.exit(1)

    print(f"Found {len(config_paths)} configs")

    for config_path in config_paths:
        row = run_config(config_path, project_root, experiments_dir, main_script)
        append_record(output_csv, row)

        print(
            f"[{row['status']}] {row['config_file']} -> "
            f"Acc={row['mean_accuracy']}, F1={row['mean_f1']}, "
            f"Precision={row['mean_precision']}, Recall={row['mean_recall']}"
        )

        if row["status"] != "success" and args.stop_on_error:
            print("Stop on error enabled, aborting batch.")
            sys.exit(1)

    print(f"\nDone. Records written to: {output_csv}")


if __name__ == "__main__":
    main()
