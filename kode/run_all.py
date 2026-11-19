"""
=============================================================================
UTS MACHINE LEARNING - MASTER REPRODUCTION SCRIPT (ONE-COMMAND REPRODUCTION)
Authors: Fransisco Sitepu (101032300004) & Ahmad Ghazali El-Faruqi HSB (101032330080)
=============================================================================
This script runs the entire machine learning pipeline in sequence:
- Task 1: Regression (Delivery Time Estimation: P50 & P90 with Conformal Calibration)
- Task 2: Classification (Fraud Detection: leakage/label-lag audit, behavioral features, calibration & cost rule)
- Task 3: Clustering (Customer Segmentation: CLR Preprocessing & K-Means K=5)
=============================================================================
"""

import os
import sys
import time

from task1_regression import run_task1
from task2_classification import run_task2
from task3_clustering import run_task3
from laporan_task2 import run as build_report_task2

def main():
    print("=" * 80)
    print("UTS MACHINE LEARNING - PIPELINE EXECUTION")
    print("Kelompok 16: Fransisco Sitepu (101032300004) & Ahmad Ghazali El-Faruqi HSB (101032330080)")
    print("=" * 80)

    start_total = time.time()

    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    # Check for dataset directory
    candidate_data_dirs = [
        os.path.abspath(os.path.join(repo_root, "..", "TUGAS MACHINE LEARNING", "data_uts_ml-20261119T075355Z-1-001", "data_uts_ml")),
        os.path.abspath(os.path.join(repo_root, "data")),
        os.path.abspath(os.path.join(repo_root, "..", "data_uts_ml")),
        os.path.abspath("data_uts_ml")
    ]

    data_dir = None
    for p in candidate_data_dirs:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "regresi_train.csv.gz")):
            data_dir = p
            break

    if data_dir is None:
        raise FileNotFoundError(f"Cannot locate UTS datasets in any of: {candidate_data_dirs}")

    print(f"Dataset directory resolved: {data_dir}")
    print(f"Output directory resolved:  {repo_root}\n")

    # Run Task 1
    t0 = time.time()
    print(">>> RUNNING TASK 1: REGRESSION (DELIVERY TIME ESTIMATION)...")
    run_task1(data_dir=data_dir, output_dir=repo_root)
    print(f">>> Task 1 completed in {time.time() - t0:.1f}s\n")

    # Run Task 2
    t0 = time.time()
    print(">>> RUNNING TASK 2: CLASSIFICATION (FRAUD DETECTION)...")
    run_task2(data_dir=data_dir, output_dir=repo_root)
    build_report_task2(os.path.join(repo_root, "laporan.pdf"))  # perbarui bagian Task 2 di laporan.pdf
    print(f">>> Task 2 completed in {time.time() - t0:.1f}s\n")

    # Run Task 3
    t0 = time.time()
    print(">>> RUNNING TASK 3: CLUSTERING (CUSTOMER SEGMENTATION)...")
    run_task3(data_dir=data_dir, output_dir=repo_root)
    print(f">>> Task 3 completed in {time.time() - t0:.1f}s\n")

    print("=" * 80)
    print(f"ALL PIPELINES EXECUTED SUCCESSFULLY in {time.time() - start_total:.1f}s!")
    print("Generated files:")
    print("  1. prediksi_regresi.csv")
    print("  2. prediksi_klasifikasi.csv")
    print("  3. cluster.csv")
    print("=" * 80)

if __name__ == '__main__':
    main()
