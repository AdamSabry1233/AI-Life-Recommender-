"""
Central test delegator.

A thin wrapper over pytest that translates friendly subset names to the
appropriate pytest path + marker selectors, and offers two distinct rerun modes:

  --watch    filesystem watcher (pytest-watcher) — reruns on file save
  --changed  git-diff-based — runs only tests whose files changed vs. base branch

Usage:
  python tools/test_runner.py                      # all
  python tools/test_runner.py fast                 # excludes @pytest.mark.slow
  python tools/test_runner.py agents               # any test tagged @agents
  python tools/test_runner.py habit                # habit agent only (path-scoped)
  python tools/test_runner.py engine | recommender | memory | integration | baselines
  python tools/test_runner.py --list               # print subset names
  python tools/test_runner.py --markers            # print pytest markers
  python tools/test_runner.py agents --watch
  python tools/test_runner.py --changed            # against origin/master
  python tools/test_runner.py --changed --base main
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"

# Subset name -> list of pytest CLI args.
# Marker-based selectors use `-m`; path-based use a relative path.
SUBSETS: dict[str, list[str]] = {
    "all":         [],
    "fast":        ["-m", "not slow"],
    "unit":        ["-m", "unit"],
    "integration": ["-m", "integration"],
    "agents":      ["-m", "agents"],
    "engine":      ["-m", "engine"],
    "recommender": ["-m", "recommender"],
    "memory":      ["-m", "memory"],
    "baselines":   ["-m", "baselines"],
    # Path-scoped single-agent targets
    "habit":           ["tests/unit/agents/test_habit_agent.py"],
    "entertainment":   ["tests/unit/agents/test_entertainment_agent.py"],
    "food":            ["tests/unit/agents/test_food_agent.py"],
    "learning":        ["tests/unit/agents/test_learning_agent.py"],
    "orchestrator":    ["tests/unit/agents/test_orchestrator.py"],
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="test_runner",
        description="Central test delegator for the AI Life Recommender project.",
    )
    p.add_argument(
        "subset",
        nargs="?",
        default="all",
        help=f"Which tests to run. One of: {', '.join(sorted(SUBSETS))}. Default: all.",
    )
    p.add_argument("--watch", action="store_true", help="Filesystem auto-rerun (pytest-watcher).")
    p.add_argument(
        "--changed",
        action="store_true",
        help="Only tests whose source files changed vs. --base.",
    )
    p.add_argument("--base", default="origin/master", help="Base ref for --changed. Default origin/master.")
    p.add_argument("--list", action="store_true", help="Print subset names and exit.")
    p.add_argument("--markers", action="store_true", help="Print pytest markers and exit.")
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to pytest (after --).",
    )
    return p


def _list_subsets() -> int:
    width = max(len(n) for n in SUBSETS)
    for name in sorted(SUBSETS):
        args = " ".join(SUBSETS[name]) or "(no filter)"
        print(f"  {name:<{width}}   {args}")
    return 0


def _print_markers() -> int:
    return subprocess.call(["pytest", "--markers"], cwd=REPO_ROOT)


def _changed_test_paths(base: str) -> list[str]:
    """
    Return test files whose source path is implicated by the diff against `base`.

    Heuristic:
      - Any changed test file is included directly.
      - Any changed src/ file maps to its mirrored tests/unit/ path if present.
      - If no mapping found, fall back to running the whole suite.
    """
    try:
        diff = subprocess.check_output(
            ["git", "diff", "--name-only", base, "--"],
            cwd=REPO_ROOT,
            text=True,
        )
    except subprocess.CalledProcessError:
        print(f"! git diff against {base} failed — running full suite.", file=sys.stderr)
        return []

    changed = [Path(p) for p in diff.splitlines() if p.strip()]
    if not changed:
        print("No changes detected — running full suite.", file=sys.stderr)
        return []

    selected: set[Path] = set()
    for f in changed:
        if f.parts and f.parts[0] == "tests" and f.suffix == ".py":
            selected.add(f)
            continue
        if f.parts and f.parts[0] == "src" and f.suffix == ".py":
            # src/agents/habit_agent.py -> tests/unit/agents/test_habit_agent.py
            stem = f.stem
            mirror = Path("tests/unit", *f.parts[1:-1], f"test_{stem}.py")
            if (REPO_ROOT / mirror).exists():
                selected.add(mirror)

    if not selected:
        print("No mapped tests for the changed files — running full suite.", file=sys.stderr)
        return []

    return [str(p) for p in sorted(selected)]


def _resolve_selector(subset: str) -> list[str]:
    if subset not in SUBSETS:
        known = ", ".join(sorted(SUBSETS))
        raise SystemExit(f"Unknown subset '{subset}'. Known: {known}")
    return list(SUBSETS[subset])


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list:
        return _list_subsets()
    if args.markers:
        return _print_markers()

    if args.changed:
        paths = _changed_test_paths(args.base)
        selector = paths if paths else []
    else:
        selector = _resolve_selector(args.subset)

    extra = [a for a in args.extra if a != "--"]

    if args.watch:
        cmd = ["pytest-watcher", str(TESTS_DIR), "--", *selector, *extra]
    else:
        cmd = ["pytest", *selector, *extra]

    print(f"> {' '.join(cmd)}", file=sys.stderr)
    return subprocess.call(cmd, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
