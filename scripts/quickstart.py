#!/usr/bin/env python3
"""
TaiYiYuan (太医院) — Quickstart

One-command bootstrap: check deps → init DB → generate demo data → plots → dashboard.

Usage:
    python scripts/quickstart.py
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "taiyiyuan.db"
PLOTS_DIR = PROJECT_ROOT / "docs" / "example-outputs"

os.environ.setdefault("TAIYIYUAN_DB", str(DB_PATH))


def _run(cmd: list[str], label: str) -> bool:
    """Run a command and report success/failure."""
    print(f"  → {label}...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"    ✗ Failed: {result.stderr[:200]}")
        return False
    return True


def check_dependencies() -> bool:
    """Check that required Python packages are installed."""
    print("\n1. Checking dependencies...")
    required = ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "httpx"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"    ✓ {pkg}")
        except ImportError:
            print(f"    ✗ {pkg} — NOT FOUND")
            missing.append(pkg)

    if missing:
        print(f"\n   Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        print(f"   Or: pip install -e {PROJECT_ROOT}")
        return False
    return True


def init_database() -> bool:
    """Initialize the SQLite database."""
    print("\n2. Initializing database...")
    if DB_PATH.exists():
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        count = conn.execute("SELECT COUNT(*) FROM body_metrics").fetchone()[0]
        conn.close()
        if count > 0:
            print(f"    ✓ Database exists with {count} body metric rows — skipping init")
            return True

    return _run([sys.executable, str(PROJECT_ROOT / "scripts" / "setup.py")],
                "Running setup.py")


def generate_demo_data() -> bool:
    """Generate 90 days of demo data."""
    print("\n3. Generating demo data...")
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    count = conn.execute("SELECT COUNT(*) FROM body_metrics").fetchone()[0]
    conn.close()

    if count > 100:
        print(f"    ✓ Demo data already present ({count} body metrics) — skipping")
        return True

    return _run([sys.executable, str(PROJECT_ROOT / "scripts" / "generate_demo_data.py")],
                "Generating 90 days of synthetic health data")


def generate_example_plots() -> bool:
    """Generate all example plots."""
    print("\n4. Generating example plots...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plots_script = str(PROJECT_ROOT / "modeling" / "plots.py")
    env = os.environ.copy()
    env["TAIYIYUAN_DB"] = str(DB_PATH)

    commands = [
        ([sys.executable, plots_script, "trend", "--metric", "weight", "--days", "90",
          "--output", str(PLOTS_DIR / "trend_weight.png")], "Weight trend"),
        ([sys.executable, plots_script, "anomalies", "--metric", "hrv", "--days", "120",
          "--output", str(PLOTS_DIR / "anomalies_hrv.png")], "HRV anomalies"),
        ([sys.executable, plots_script, "power", "--metric", "sleep_quality", "--baseline_days", "60",
          "--output", str(PLOTS_DIR / "power_sleep.png")], "Power analysis"),
        ([sys.executable, plots_script, "forest", "--trial_id", "1",
          "--output", str(PLOTS_DIR / "forest_trial.png")], "Trial forest plot"),
        ([sys.executable, plots_script, "heatmap", "--days", "90",
          "--output", str(PLOTS_DIR / "correlations.png")], "Correlation heatmap"),
        ([sys.executable, plots_script, "network", "--days", "90",
          "--output", str(PLOTS_DIR / "network.png")], "Network graph"),
    ]

    success = 0
    for cmd, label in commands:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                cwd=str(PROJECT_ROOT))
        if result.returncode == 0:
            print(f"    ✓ {label}")
            success += 1
        else:
            print(f"    ✗ {label}: {result.stderr[:100]}")

    print(f"\n    Generated {success}/{len(commands)} plots in {PLOTS_DIR}")
    return success > 0


def start_dashboard() -> bool:
    """Start the dashboard server."""
    print("\n5. Starting dashboard...")
    dashboard = PROJECT_ROOT / "dashboard" / "server.py"
    if not dashboard.exists():
        print("    ✗ Dashboard not found")
        return False

    print("    Starting at http://localhost:8420")
    print("    (Press Ctrl+C to stop)\n")
    env = os.environ.copy()
    env["TAIYIYUAN_DB"] = str(DB_PATH)
    try:
        subprocess.Popen([sys.executable, str(dashboard)], env=env, cwd=str(PROJECT_ROOT))
        print("    ✓ Dashboard running at http://localhost:8420")
        return True
    except Exception as e:
        print(f"    ✗ Failed to start: {e}")
        return False


def main():
    print("=" * 60)
    print("太医院 (TaiYiYuan) — Longevity OS Quickstart")
    print("=" * 60)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Database:     {DB_PATH}")

    if not check_dependencies():
        sys.exit(1)

    if not init_database():
        sys.exit(1)

    if not generate_demo_data():
        print("    ⚠ Demo data generation failed — continuing without it")

    generate_example_plots()

    print("\n" + "=" * 60)
    print("✓ Quickstart complete!")
    print("=" * 60)
    print(f"""
What you can do now:

  # Generate plots from the demo data
  TAIYIYUAN_DB={DB_PATH} python modeling/plots.py trend --metric weight --days 90 --output my_plot.png

  # Run the full trial analysis
  TAIYIYUAN_DB={DB_PATH} python modeling/causal.py analyze_trial --trial_id 1

  # Start the interactive dashboard
  TAIYIYUAN_DB={DB_PATH} python dashboard/server.py
  # Then open http://localhost:8420

  # Explore correlations
  TAIYIYUAN_DB={DB_PATH} python modeling/patterns.py scan --days 90

For more, see README.md
""")


if __name__ == "__main__":
    main()
