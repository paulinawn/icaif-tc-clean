# Example deployment for the refactored layout

A template showing how the refactored code would be scheduled with systemd. It is **not** the configuration that ran during the competition; that is preserved unchanged in `../systemd/`.

| File | Purpose |
| --- | --- |
| `icaif-agent.service` | Runs one decision cycle with `agent/live/orchestrator.py`, sandboxed, as an unprivileged user |
| `icaif-agent.timer` | Starts the service seven times per trading day, shortly before each round deadline |

Differences from the original:
- Role-based paths (`agent/live/orchestrator.py`) instead of versioned release folders
- Credentials loaded from an `EnvironmentFile` outside the repository
- Dry-run (`--get-only`) as the default; live mode must be enabled deliberately
- No competition-specific dates in the timer

**Before this could run:** the live modules still contain the competition server's paths (`/opt/icaif2026/...`). Those need to move into configuration first; see the next steps in `POC_REPORT.md`.

To install on a Linux server (as root):

```bash
cp icaif-agent.service icaif-agent.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now icaif-agent.timer
systemctl list-timers icaif-agent.timer
```