"""
run_all.py  –  Full pipeline runner
Market Basket Analysis | MSc Data Science – Data Mining
Run from the project root: python run_all.py
"""

import subprocess
import sys
import time

SCRIPTS = [
    ("src/01_data_preprocessing.py", "Data Preprocessing"),
    ("src/02_eda.py",                "Exploratory Data Analysis"),
    ("src/03_apriori_mining.py",     "Apriori Rule Mining"),
    ("src/04_rule_analysis.py",      "Rule Analysis & Interpretation"),
    ("src/05_visualisation.py",      "Visualisation"),
]

def run(script, label):
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - t0
    status = "✓ OK" if result.returncode == 0 else "✗ FAILED"
    print(f"\n[{status}]  {label}  ({elapsed:.1f}s)")
    if result.returncode != 0:
        sys.exit(result.returncode)

if __name__ == "__main__":
    print("Market Basket Analysis – Full Pipeline")
    print("MSc Data Science | Data Mining Assignment\n")
    total_start = time.time()
    for script, label in SCRIPTS:
        run(script, label)
    print(f"\n{'='*60}")
    print(f"  ALL STEPS COMPLETE  ({time.time()-total_start:.0f}s total)")
    print(f"  Outputs written to: outputs/")
    print(f"{'='*60}")
