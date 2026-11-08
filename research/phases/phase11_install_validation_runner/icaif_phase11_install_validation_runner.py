from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ENTRY_SOURCE = 'from __future__ import annotations\n\nfrom collections.abc import Mapping\nfrom typing import Any\n\nfrom our_agent.strategies.validation_baseline import (\n    ACTION_REBALANCE,\n    STRATEGY_VERSION,\n    decide,\n)\n\nEXPECTED_PHASE = "validation"\nEXPECTED_ROUND_ID = "validation-2026-10-08-r1"\n\n\nclass LiveEntryGuardError(RuntimeError):\n    pass\n\n\ndef _get(value: Any, *names: str) -> Any:\n    if isinstance(value, Mapping):\n        for name in names:\n            if name in value:\n                return value[name]\n    for name in names:\n        if hasattr(value, name):\n            return getattr(value, name)\n    return None\n\n\ndef _looks_like_round(value: Any) -> bool:\n    return (\n        _get(value, "round_id", "roundId", "id") is not None\n        and _get(value, "phase") is not None\n    )\n\n\ndef _looks_like_portfolio(value: Any) -> bool:\n    markers = (\n        "positions", "holdings", "cash", "nav", "weights",\n        "last_decision", "lastDecision",\n    )\n    return any(_get(value, marker) is not None for marker in markers)\n\n\ndef _extract_context(args, kwargs):\n    round_info = None\n    portfolio = None\n\n    for key in ("round_info", "round", "current_round", "currentRound"):\n        value = kwargs.get(key)\n        if value is not None and _looks_like_round(value):\n            round_info = value\n            break\n\n    for key in ("portfolio", "current_portfolio", "currentPortfolio"):\n        value = kwargs.get(key)\n        if value is not None and _looks_like_portfolio(value):\n            portfolio = value\n            break\n\n    for value in args:\n        if round_info is None and _looks_like_round(value):\n            round_info = value\n            continue\n        if portfolio is None and _looks_like_portfolio(value):\n            portfolio = value\n\n    if round_info is None:\n        raise LiveEntryGuardError(\n            "Watcher did not provide a recognizable round object."\n        )\n    if portfolio is None:\n        raise LiveEntryGuardError(\n            "Watcher did not provide a recognizable portfolio object."\n        )\n\n    return round_info, portfolio\n\n\ndef strategy(*args, **kwargs):\n    """\n    Return weights only for Validation day 1 / Round 1 and only for\n    a clearly fresh all-cash portfolio. Every other state fails closed.\n    """\n    round_info, portfolio = _extract_context(args, kwargs)\n\n    phase = str(_get(round_info, "phase") or "").strip().lower()\n    round_id = str(\n        _get(round_info, "round_id", "roundId", "id") or ""\n    ).strip()\n\n    if phase != EXPECTED_PHASE:\n        raise LiveEntryGuardError(\n            f"Refusing decision outside Validation: {phase!r}."\n        )\n\n    if round_id != EXPECTED_ROUND_ID:\n        raise LiveEntryGuardError(\n            f"Refusing unexpected round {round_id!r}; "\n            f"expected {EXPECTED_ROUND_ID!r}."\n        )\n\n    decision = decide(portfolio)\n\n    if decision.action != ACTION_REBALANCE or decision.weights is None:\n        raise LiveEntryGuardError(\n            "Frozen baseline does not request initial deployment. "\n            "Refusing a second or ambiguous submission."\n        )\n\n    return decision.weights\n\n\ndef metadata():\n    return {\n        "adapter": "validation-r1-one-shot-v1",\n        "strategy_version": STRATEGY_VERSION,\n        "expected_phase": EXPECTED_PHASE,\n        "expected_round_id": EXPECTED_ROUND_ID,\n        "after_r1": "process exits; later missing decisions intentionally HOLD",\n    }\n'
RUNNER_SOURCE = 'from __future__ import annotations\n\nimport argparse\nimport os\nimport subprocess\nimport sys\nfrom pathlib import Path\n\nSTRATEGY_PATH = "our_agent.strategies.validation_entry:strategy"\nLOCK_NAME = "validation-r1-single-writer.lock"\n\n\ndef find_project_root() -> Path:\n    for candidate in [Path.cwd(), *Path.cwd().parents]:\n        if (\n            (candidate / "universe.json").exists()\n            and (candidate / "tools" / "auto_submit.py").exists()\n            and (candidate / "our_agent").is_dir()\n        ):\n            return candidate.resolve()\n    raise SystemExit(\n        "ERROR: Run this from the starter-kit directory."\n    )\n\n\nROOT = find_project_root()\nLOCK = ROOT / ".icaif" / LOCK_NAME\n\n\ndef call_official(*args: str) -> int:\n    command = [sys.executable, "tools/auto_submit.py", *args]\n    print(">", " ".join(command))\n    return subprocess.run(\n        command,\n        cwd=ROOT,\n        text=True,\n    ).returncode\n\n\ndef read_only_preflight() -> None:\n    print("=== READ-ONLY VALIDATION PREFLIGHT ===")\n\n    checks = [\n        ("schedule",),\n        ("portfolio", "--phase", "validation"),\n        ("decisions", "--phase", "validation"),\n    ]\n\n    for args in checks:\n        if call_official(*args) != 0:\n            raise SystemExit(\n                "ERROR: Read-only preflight command failed: "\n                + " ".join(args)\n            )\n\n    print("READ-ONLY PREFLIGHT COMPLETE.")\n    print("No competition submission was made.")\n\n\ndef acquire_lock() -> int:\n    LOCK.parent.mkdir(parents=True, exist_ok=True)\n    try:\n        fd = os.open(\n            LOCK,\n            os.O_CREAT | os.O_EXCL | os.O_WRONLY,\n        )\n    except FileExistsError:\n        raise SystemExit(\n            "ERROR: Single-writer lock already exists. "\n            "Do NOT delete it casually; inspect prior submission state first."\n        )\n\n    os.write(fd, b"Validation R1 one-shot writer lock.\\n")\n    os.fsync(fd)\n    return fd\n\n\ndef arm_validation_r1() -> None:\n    print("=== LIVE VALIDATION R1 ONE-SHOT ===")\n    print(\n        "WARNING: this mode may create the irreversible first "\n        "Validation decision."\n    )\n\n    fd = acquire_lock()\n    try:\n        command = [\n            sys.executable,\n            "tools/auto_submit.py",\n            "watch",\n            "--phase",\n            "validation",\n            "--strategy",\n            STRATEGY_PATH,\n            "--once",\n        ]\n        print(">", " ".join(command))\n        completed = subprocess.run(\n            command,\n            cwd=ROOT,\n            text=True,\n        )\n        if completed.returncode != 0:\n            raise SystemExit(\n                "ERROR: official watch exited non-zero. KEEP the lock "\n                "and use read-only status checks before any retry."\n            )\n\n        print("Official one-shot watcher exited successfully.")\n        print("KEEP the lock. Do not run live mode again.")\n    finally:\n        os.close(fd)\n\n\ndef main() -> None:\n    parser = argparse.ArgumentParser()\n    parser.add_argument(\n        "--arm-validation-r1",\n        action="store_true",\n        help=(\n            "Explicitly allow the one-shot Validation R1 watcher. "\n            "Without this flag the runner is read-only."\n        ),\n    )\n    args = parser.parse_args()\n\n    if args.arm_validation_r1:\n        arm_validation_r1()\n    else:\n        read_only_preflight()\n\n\nif __name__ == "__main__":\n    main()\n'
TEST_SOURCE = 'from __future__ import annotations\n\nimport math\nimport unittest\n\nfrom our_agent.strategies.validation_entry import (\n    EXPECTED_ROUND_ID,\n    LiveEntryGuardError,\n    strategy,\n)\n\n\ndef fresh():\n    return {\n        "cash": 1_000_000.0,\n        "nav": 1_000_000.0,\n        "positions": [],\n        "last_decision": None,\n    }\n\n\nclass ValidationEntryTests(unittest.TestCase):\n    def test_exact_r1_deploys(self):\n        weights = strategy(\n            {"phase": "validation", "round_id": EXPECTED_ROUND_ID},\n            fresh(),\n        )\n        self.assertEqual(len(weights), 30)\n        self.assertTrue(\n            math.isclose(\n                math.fsum(weights.values()),\n                0.80,\n                abs_tol=1e-12,\n            )\n        )\n\n    def test_keyword_contract(self):\n        weights = strategy(\n            current_round={\n                "phase": "validation",\n                "round_id": EXPECTED_ROUND_ID,\n            },\n            portfolio=fresh(),\n        )\n        self.assertEqual(len(weights), 30)\n\n    def test_rejects_round2(self):\n        with self.assertRaises(LiveEntryGuardError):\n            strategy(\n                {\n                    "phase": "validation",\n                    "round_id": "validation-2026-10-08-r2",\n                },\n                fresh(),\n            )\n\n    def test_rejects_official(self):\n        with self.assertRaises(LiveEntryGuardError):\n            strategy(\n                {\n                    "phase": "official",\n                    "round_id": "official-2026-10-12-r1",\n                },\n                fresh(),\n            )\n\n    def test_rejects_deployed(self):\n        with self.assertRaises(LiveEntryGuardError):\n            strategy(\n                {"phase": "validation", "round_id": EXPECTED_ROUND_ID},\n                {\n                    "cash": 200_000.0,\n                    "nav": 1_000_000.0,\n                    "positions": [{"symbol": "AAPL", "shares": 1.0}],\n                },\n            )\n\n    def test_rejects_prior_decision(self):\n        with self.assertRaises(LiveEntryGuardError):\n            strategy(\n                {"phase": "validation", "round_id": EXPECTED_ROUND_ID},\n                {\n                    "cash": 1_000_000.0,\n                    "nav": 1_000_000.0,\n                    "positions": [],\n                    "last_decision": {"round_id": EXPECTED_ROUND_ID},\n                },\n            )\n\n\nif __name__ == "__main__":\n    unittest.main()\n'


def fail(message: str) -> None:
    raise SystemExit("\nERROR: " + message + "\n")


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (
            (candidate / "universe.json").exists()
            and (candidate / "our_agent").is_dir()
            and (candidate / "tools" / "auto_submit.py").exists()
        ):
            return candidate.resolve()
    fail(
        "Could not locate starter-kit. Run this from the starter-kit folder."
    )


ROOT = find_project_root()


def output(*args: str) -> str:
    return subprocess.check_output(
        args,
        cwd=ROOT,
        text=True,
    ).strip()


def run(*args: str) -> None:
    print(">", " ".join(args))
    cp = subprocess.run(args, cwd=ROOT, text=True)
    if cp.returncode != 0:
        raise SystemExit(cp.returncode)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cleanup_caches() -> None:
    for base_name in ("our_agent", "tests", "tools"):
        base = ROOT / base_name
        if not base.exists():
            continue
        for cache in base.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)


def main() -> None:
    print("=== PHASE 11: INSTALL VALIDATION ONE-SHOT RUNNER ===")
    print("Competition submission during install: DISABLED")
    print()

    branch = output("git", "branch", "--show-current")
    if branch != "our-agent":
        fail("Expected branch 'our-agent', found " + repr(branch))

    dirty = output("git", "status", "--short")
    if dirty:
        fail("Working tree is not clean:\n" + dirty)

    baseline = ROOT / "our_agent" / "strategies" / "validation_baseline.py"
    if not baseline.exists():
        fail("Frozen validation baseline is missing.")

    if "validation-equal30-80-hold-v1" not in baseline.read_text(
        encoding="utf-8"
    ):
        fail("Unexpected frozen baseline version.")

    auto_submit = ROOT / "tools" / "auto_submit.py"
    readme = ROOT / "README.md"
    rules = ROOT / "docs" / "rules.md"

    for path in (auto_submit, readme, rules):
        if not path.exists():
            fail("Missing official starter-kit file: " + str(path))

    auto_text = auto_submit.read_text(encoding="utf-8")
    for required in ("watch", "--strategy", "--once"):
        if required not in auto_text:
            fail(
                "Local auto_submit.py lacks expected watch support: "
                + required
            )

    print("Local official-file hashes:")
    print(" auto_submit.py:", sha256(auto_submit))
    print(" README.md:", sha256(readme))
    print(" rules.md:", sha256(rules))

    entry_path = ROOT / "our_agent" / "strategies" / "validation_entry.py"
    runner_path = ROOT / "tools" / "run_validation_baseline.py"
    test_path = ROOT / "tests" / "test_validation_entry.py"

    compile(ENTRY_SOURCE, str(entry_path), "exec")
    compile(RUNNER_SOURCE, str(runner_path), "exec")
    compile(TEST_SOURCE, str(test_path), "exec")

    entry_path.write_text(ENTRY_SOURCE, encoding="utf-8")
    runner_path.write_text(RUNNER_SOURCE, encoding="utf-8")
    test_path.write_text(TEST_SOURCE, encoding="utf-8")

    print()
    print("=== ENTRY GUARD TESTS ===")
    run(
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_validation_entry.py",
        "-v",
    )

    print()
    print("=== FULL REGRESSION TESTS ===")
    run(
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-v",
    )

    print()
    print("=== OFFICIAL WATCH CLI CHECK ===")
    cp = subprocess.run(
        [
            sys.executable,
            "tools/auto_submit.py",
            "watch",
            "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if cp.returncode != 0:
        fail("Official watch --help failed.")

    for required in ("--phase", "--strategy", "--once"):
        if required not in cp.stdout:
            fail("Official watch --help missing " + required)

    print("PASS: --phase / --strategy / --once")

    print()
    print("=== STATIC CHECKS ===")
    run(
        sys.executable,
        "-m",
        "compileall",
        "-q",
        "our_agent",
        "tests",
        "tools/run_validation_baseline.py",
    )
    cleanup_caches()
    run("git", "diff", "--check")

    changelog = ROOT / "CHANGELOG.md"
    marker = "## 2026-10-07 — Validation one-shot runner installed"
    current = changelog.read_text(encoding="utf-8")
    if marker not in current:
        if current and not current.endswith("\n"):
            current += "\n"
        current += (
            "\n"
            + marker
            + "\n\n"
            + "- Added one-shot Validation R1 adapter with hard phase/round/fresh-portfolio guards.\n"
            + "- Added read-only-by-default Validation runner.\n"
            + "- Live mode requires explicit `--arm-validation-r1`.\n"
            + "- Added `.icaif` single-writer lock for live R1 invocation.\n"
            + "- After the one-shot R1 process exits, later missing decisions intentionally HOLD.\n"
            + "- No competition submission was made during installation.\n"
        )
        changelog.write_text(current, encoding="utf-8")

    print()
    print("=== GIT CHECKPOINT ===")
    run(
        "git",
        "add",
        "our_agent/strategies/validation_entry.py",
        "tools/run_validation_baseline.py",
        "tests/test_validation_entry.py",
        "CHANGELOG.md",
    )
    run("git", "diff", "--cached", "--check")
    run(
        "git",
        "commit",
        "-m",
        "Install guarded Validation one-shot runner",
    )

    cleanup_caches()
    final_status = output("git", "status", "--short")
    if final_status:
        fail("Working tree is not clean after commit:\n" + final_status)

    print()
    print("=== READ-ONLY LIVE PREFLIGHT ===")
    cp = subprocess.run(
        [sys.executable, "tools/run_validation_baseline.py"],
        cwd=ROOT,
        text=True,
    )
    if cp.returncode != 0:
        fail(
            "Read-only live preflight failed. "
            "Runner is installed; no submission was made."
        )

    print()
    print("=== PHASE 11 COMPLETE ===")
    print(
        "Strategy adapter: "
        "our_agent.strategies.validation_entry:strategy"
    )
    print("Runner: tools/run_validation_baseline.py")
    print("Default runner mode: READ-ONLY")
    print("Live flag: --arm-validation-r1")
    print("No competition submission was made.")
    print("HEAD:", output("git", "rev-parse", "HEAD"))
    print("Working tree: CLEAN")


if __name__ == "__main__":
    main()
