import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODULES = [
    "config.py",
    "macro_calendar.py",
    "data_fetcher.py",
    "corporate_events_fetcher.py",
    "news_sentiment_fetcher.py",
    "event_classifier.py",
    "global_risk_monitor.py",
    "feature_engineer.py",
    "model_trainer.py",
    "ensemble_manager.py",
    "runtime_validator.py",
    "predictor.py",
    "human_insight_manager.py",
    "history_manager.py",
    "backtester.py",
    "scheduler.py",
]
def run_module(module_name: str):
    """Run a module and return whether it passed."""

    print(f"\n{'=' * 60}")
    print(f"Running: {module_name}")
    print(f"{'=' * 60}")

    start_time = time.perf_counter()

    result = subprocess.run(
        [sys.executable, module_name],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    elapsed = time.perf_counter() - start_time

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    passed = "STATUS: PASS" in result.stdout

    return passed, elapsed

def main():
    total = len(MODULES)
    passed = 0
    failed = []

    print("\nStarting Project Self-Test Automation...\n")

    for module in MODULES:
     passed_test, elapsed = run_module(module)

     if passed_test:
        print(f"✅ {module} PASSED ({elapsed:.2f} sec)")
        passed += 1
     else:
        print(f"❌ {module} FAILED ({elapsed:.2f} sec)")
        failed.append(module)

    print("\n" + "=" * 60)
    print("SELF-TEST SUMMARY")
    print("=" * 60)
    print(f"Passed : {passed}/{total}")
    print(f"Failed : {total - passed}/{total}")

    if failed:
        print("\nFailed modules:")
        for module in failed:
            print(f" - {module}")
    else:
        print("\n🎉 All modules passed successfully!")

if __name__ == "__main__":
    main()