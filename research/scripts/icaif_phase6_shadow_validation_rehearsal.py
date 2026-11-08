from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
ARTIFACTS = ROOT / "artifacts"
REPORT = ARTIFACTS / "validation_shadow_rehearsal.json"


def run_capture(*args: str) -> tuple[int, str, str]:
    cp = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return cp.returncode, cp.stdout, cp.stderr


def run_stream(*args: str) -> None:
    print(">", " ".join(args))
    cp = subprocess.run(args, cwd=ROOT, text=True)
    if cp.returncode != 0:
        raise SystemExit(cp.returncode)


def output(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


required = [
    ROOT / "universe.json",
    ROOT / "tools" / "auto_submit.py",
    ROOT / "tools" / "dry_run_agent.py",
    ROOT / "tools" / "preflight_dry_run.py",
    ROOT / "our_agent",
    ROOT / ".gitignore",
]
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    raise SystemExit(
        "ERROR: Run from starter-kit. Missing: " + repr(missing)
    )

branch = output("git", "branch", "--show-current")
if branch != "our-agent":
    raise SystemExit(
        f"ERROR: Expected branch 'our-agent', found '{branch}'."
    )

if subprocess.run(
    ["git", "check-ignore", "-q", ".env"],
    cwd=ROOT,
).returncode != 0:
    raise SystemExit("ERROR: .env must remain Git-ignored.")

if subprocess.run(
    ["git", "check-ignore", "-q", ".icaif/credentials.json"],
    cwd=ROOT,
).returncode != 0:
    raise SystemExit(
        "ERROR: .icaif/credentials.json must remain Git-ignored."
    )

print("=== ICAIF PHASE 6: VALIDATION SHADOW REHEARSAL ===")
print("Competition submission: DISABLED")
print("All organizer calls in this script are GET/read-only.")
print()

print("=== LOCAL REGRESSION SUITE ===")
run_stream(
    sys.executable,
    "-m",
    "unittest",
    "discover",
    "-s",
    "tests",
    "-v",
)
run_stream(
    sys.executable,
    "-m",
    "compileall",
    "-q",
    "our_agent",
    "tests",
    "tools",
)
run_stream("git", "diff", "--check")

print()
print("=== LIVE SCHEDULE ===")
schedule_rc, schedule_out, schedule_err = run_capture(
    sys.executable,
    "tools/auto_submit.py",
    "schedule",
)
print(schedule_out.rstrip())
if schedule_rc != 0:
    print(schedule_err.rstrip())
    raise SystemExit(schedule_rc)

print()
print("=== VALIDATION PORTFOLIO (READ-ONLY) ===")
portfolio_rc, portfolio_out, portfolio_err = run_capture(
    sys.executable,
    "tools/auto_submit.py",
    "portfolio",
    "--phase",
    "validation",
)
print(portfolio_out.rstrip())
if portfolio_rc != 0:
    print(portfolio_err.rstrip())
    raise SystemExit(portfolio_rc)

print()
print("=== REFRESH CANDIDATE FROM ALPACA ===")
run_stream(
    sys.executable,
    "tools/dry_run_agent.py",
)

print()
print("=== PREFLIGHT DRY RUN ===")
run_stream(
    sys.executable,
    "tools/preflight_dry_run.py",
)

candidate = ARTIFACTS / "candidate_v0_dry_run.json"
alpaca_diag = ARTIFACTS / "alpaca_smoke_test.json"

if not candidate.exists():
    raise SystemExit(
        "ERROR: candidate_v0_dry_run.json was not produced."
    )

report = {
    "artifact_type": "VALIDATION_SHADOW_REHEARSAL_NO_SUBMISSION",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "git_branch": branch,
    "git_head": output("git", "rev-parse", "HEAD"),
    "submission_enabled": False,
    "organizer_calls": [
        "schedule GET/read-only",
        "portfolio GET/read-only",
    ],
    "candidate_artifact": {
        "path": str(candidate),
        "sha256": sha256(candidate),
    },
    "schedule_output_captured": True,
    "portfolio_output_captured": True,
}

if alpaca_diag.exists():
    report["alpaca_diagnostic"] = {
        "path": str(alpaca_diag),
        "sha256": sha256(alpaca_diag),
    }

ARTIFACTS.mkdir(parents=True, exist_ok=True)
REPORT.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print()
print("=== REHEARSAL REPORT ===")
print(f"Report: {REPORT}")
print(f"Candidate SHA256: {report['candidate_artifact']['sha256']}")
print("Contains team token: NO")
print("Contains Alpaca secret: NO")
print("Competition submission made: NO")

print()
print("=== FINAL GIT STATUS ===")
run_stream("git", "status", "--short")

print()
print("PHASE 6 SHADOW REHEARSAL COMPLETE.")
print("No competition submission was made.")
