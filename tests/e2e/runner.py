"""Seizure-Sentry Opaque-Box E2E Test Suite Runner.

Single-command CLI test runner for Windows and Linux environments.
Discovers and executes tests across all tiers (Tier 1-4 + Smoke), collects
metrics, and produces formatted summary reports with exit code 0 or 1.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.e2e.config import CONFIG, set_config


class E2ETestResult(unittest.TextTestResult):
    """Custom test result tracking test outcomes by tier category."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.tier_stats: Dict[str, Dict[str, int]] = {
            "Harness / Smoke": {"total": 0, "pass": 0, "fail": 0, "error": 0, "skip": 0},
            "Tier 1: Features": {"total": 0, "pass": 0, "fail": 0, "error": 0, "skip": 0},
            "Tier 2: Boundaries": {"total": 0, "pass": 0, "fail": 0, "error": 0, "skip": 0},
            "Tier 3: Combinations": {"total": 0, "pass": 0, "fail": 0, "error": 0, "skip": 0},
            "Tier 4: Scenarios": {"total": 0, "pass": 0, "fail": 0, "error": 0, "skip": 0},
            "Other": {"total": 0, "pass": 0, "fail": 0, "error": 0, "skip": 0},
        }
        self.tier_durations: Dict[str, float] = {k: 0.0 for k in self.tier_stats}
        self._test_start_time = 0.0

    def _get_tier_for_test(self, test: unittest.TestCase) -> str:
        mod_name = test.__class__.__module__
        if "tier1" in mod_name:
            return "Tier 1: Features"
        if "tier2" in mod_name:
            return "Tier 2: Boundaries"
        if "tier3" in mod_name:
            return "Tier 3: Combinations"
        if "tier4" in mod_name:
            return "Tier 4: Scenarios"
        if "smoke" in mod_name or "test_smoke" in str(test):
            return "Harness / Smoke"
        return "Other"

    def startTest(self, test: unittest.TestCase):
        super().startTest(test)
        self._test_start_time = time.perf_counter()
        tier = self._get_tier_for_test(test)
        self.tier_stats[tier]["total"] += 1

    def addSuccess(self, test: unittest.TestCase):
        super().addSuccess(test)
        tier = self._get_tier_for_test(test)
        self.tier_stats[tier]["pass"] += 1
        self.tier_durations[tier] += time.perf_counter() - self._test_start_time

    def addFailure(self, test: unittest.TestCase, err):
        super().addFailure(test, err)
        tier = self._get_tier_for_test(test)
        self.tier_stats[tier]["fail"] += 1
        self.tier_durations[tier] += time.perf_counter() - self._test_start_time

    def addError(self, test: unittest.TestCase, err):
        super().addError(test, err)
        tier = self._get_tier_for_test(test)
        self.tier_stats[tier]["error"] += 1
        self.tier_durations[tier] += time.perf_counter() - self._test_start_time

    def addSkip(self, test: unittest.TestCase, reason: str):
        super().addSkip(test, reason)
        tier = self._get_tier_for_test(test)
        self.tier_stats[tier]["skip"] += 1
        self.tier_durations[tier] += time.perf_counter() - self._test_start_time


def discover_suite(e2e_dir: Path, tier_selection: str) -> unittest.TestSuite:
    """Discovers test suites matching the selected tier."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    tier_map = {
        "1": [e2e_dir / "tier1_features"],
        "2": [e2e_dir / "tier2_boundaries"],
        "3": [e2e_dir / "tier3_combinations"],
        "4": [e2e_dir / "tier4_scenarios"],
        "all": [
            e2e_dir,  # Discovers test_smoke.py and other top-level tests
            e2e_dir / "tier1_features",
            e2e_dir / "tier2_boundaries",
            e2e_dir / "tier3_combinations",
            e2e_dir / "tier4_scenarios",
        ],
    }

    targets = tier_map.get(tier_selection, [e2e_dir])

    for target in targets:
        if not target.exists():
            continue
        if target == e2e_dir:
            # Top-level e2e directory: discover non-recursive test_*.py
            for py_file in target.glob("test_*.py"):
                module_name = f"tests.e2e.{py_file.stem}"
                try:
                    loaded = loader.loadTestsFromName(module_name)
                    suite.addTests(loaded)
                except Exception as e:
                    print(f"[WARN] Failed to load {module_name}: {e}", file=sys.stderr)
        else:
            # Recursive tier discovery
            discovered = loader.discover(
                start_dir=str(target),
                pattern="test_*.py",
                top_level_dir=str(PROJECT_ROOT),
            )
            suite.addTests(discovered)

    return suite


def print_summary_table(result: E2ETestResult, wall_clock_sec: float) -> None:
    """Renders formatted ASCII summary table of test results across all tiers."""
    col_w_tier = 22
    col_w_num = 7
    col_w_time = 10

    border_line = "=" * (col_w_tier + 5 * col_w_num + col_w_time + 8)
    sub_line = "-" * (col_w_tier + 5 * col_w_num + col_w_time + 8)

    print("\n" + border_line)
    print("                      SEIZURE-SENTRY E2E TEST REPORT")
    print(border_line)
    print(
        f"{'Tier':<{col_w_tier}} | "
        f"{'Total':>{col_w_num - 2}} | "
        f"{'Pass':>{col_w_num - 2}} | "
        f"{'Fail':>{col_w_num - 2}} | "
        f"{'Error':>{col_w_num - 2}} | "
        f"{'Skip':>{col_w_num - 2}} | "
        f"{'Time':>{col_w_time - 2}}"
    )
    print(sub_line)

    tot_all = 0
    tot_pass = 0
    tot_fail = 0
    tot_error = 0
    tot_skip = 0

    for tier_name, stats in result.tier_stats.items():
        if stats["total"] == 0 and tier_name == "Other":
            continue
        tot_all += stats["total"]
        tot_pass += stats["pass"]
        tot_fail += stats["fail"]
        tot_error += stats["error"]
        tot_skip += stats["skip"]
        duration = result.tier_durations.get(tier_name, 0.0)

        print(
            f"{tier_name:<{col_w_tier}} | "
            f"{stats['total']:>{col_w_num - 2}} | "
            f"{stats['pass']:>{col_w_num - 2}} | "
            f"{stats['fail']:>{col_w_num - 2}} | "
            f"{stats['error']:>{col_w_num - 2}} | "
            f"{stats['skip']:>{col_w_num - 2}} | "
            f"{duration:>{col_w_time - 3}.2f}s"
        )

    print(sub_line)
    print(
        f"{'TOTAL':<{col_w_tier}} | "
        f"{tot_all:>{col_w_num - 2}} | "
        f"{tot_pass:>{col_w_num - 2}} | "
        f"{tot_fail:>{col_w_num - 2}} | "
        f"{tot_error:>{col_w_num - 2}} | "
        f"{tot_skip:>{col_w_num - 2}} | "
        f"{wall_clock_sec:>{col_w_time - 3}.2f}s"
    )
    print(border_line)

    if tot_fail == 0 and tot_error == 0:
        print("OVERALL STATUS: [PASS] All executed tests passed cleanly.")
    else:
        print(f"OVERALL STATUS: [FAIL] {tot_fail} failed, {tot_error} errors encountered.")
    print(border_line + "\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seizure-Sentry Opaque-Box E2E Test Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Select specific test tier to execute",
    )
    parser.add_argument(
        "--failfast",
        action="store_true",
        help="Stop immediately on first failure or error",
    )
    parser.add_argument(
        "--target-url",
        default=CONFIG.target_url,
        help="Backend base REST URL",
    )
    parser.add_argument(
        "--ws-url",
        default=CONFIG.ws_url,
        help="Backend base WebSocket URL",
    )
    parser.add_argument(
        "--access-key",
        default=CONFIG.access_key,
        help="Backend authorization access key",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed per-test output",
    )
    return parser.parse_args(argv)


def run_e2e_tests(argv: Optional[List[str]] = None) -> int:
    """Main entrypoint for test execution. Returns exit code 0 or 1."""
    args = parse_args(argv)

    # Configure global runtime settings
    set_config(
        target_url=args.target_url,
        ws_url=args.ws_url,
        access_key=args.access_key,
    )

    e2e_dir = PROJECT_ROOT / "tests" / "e2e"
    suite = discover_suite(e2e_dir, args.tier)

    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        failfast=args.failfast,
        resultclass=E2ETestResult,
    )

    start_wall = time.perf_counter()
    result = runner.run(suite)
    wall_duration = time.perf_counter() - start_wall

    print_summary_table(result, wall_duration)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_e2e_tests())
