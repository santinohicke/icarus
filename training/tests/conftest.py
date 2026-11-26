"""Shared test configuration.

With ``ICARUS_REQUIRE_ALL_TESTS=1`` (set in CI) any skipped test or module,
for example one that needs Gymnasium or Stable-Baselines3 through
``pytest.importorskip``, fails the run: CI must never silently leave the
adapters untested. Locally, skips stay skips.
"""

from __future__ import annotations

import os

import pytest

_skipped: list[str] = []


def _require_all() -> bool:
    return os.environ.get("ICARUS_REQUIRE_ALL_TESTS", "") not in ("", "0")


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.skipped:
        _skipped.append(report.nodeid)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    # An expected failure (xfail) is reported as skipped too; it is not a skip.
    if report.skipped and not hasattr(report, "wasxfail"):
        _skipped.append(report.nodeid)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _require_all() and _skipped:
        print(f"\nICARUS_REQUIRE_ALL_TESTS: {len(_skipped)} skipped: {_skipped}")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
