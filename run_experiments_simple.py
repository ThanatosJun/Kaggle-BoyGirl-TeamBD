#!/usr/bin/env python3
"""Simple script to run experiments sequentially."""

import subprocess
import sys
import time

experiments = [
    ("Method 1: Global Mean", "configs/exp1_p99_method1_global_mean.yaml"),
    ("Method 2: Paper Range", "configs/exp1_p99_method2_paper_range.yaml"),
    ("Method 3: Grouped Mean", "configs/exp1_p99_method3_grouped_mean.yaml"),
]

print("=" * 60)
print("Starting Experiment Series")
print("=" * 60)

for exp_name, config_file in experiments:
    print(f"\n>>> Running: {exp_name}")
    print(f">>> Config: {config_file}")
    print("-" * 60)

    cmd = [sys.executable, "main_train.py", "--config", config_file]

    try:
        result = subprocess.run(cmd, timeout=600)
        if result.returncode == 0:
            print(f"OK: {exp_name} completed successfully")
        else:
            print(f"ERROR: {exp_name} failed with return code {result.returncode}")
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {exp_name} took too long")
    except Exception as e:
        print(f"ERROR: {exp_name} failed: {str(e)}")

    print("-" * 60)
    time.sleep(2)

print("\n" + "=" * 60)
print("All experiments completed!")
print("=" * 60)
