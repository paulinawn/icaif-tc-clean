# Deployment: systemd configuration

How the agent was scheduled and sandboxed on the Linux server during the competition. These files are kept exactly as deployed; credentials were redacted before sharing.

## Files

| File | Purpose |
| --- | --- |
| `icaif-official-v391.timer` | Starts the service on weekdays Oct 12–30, 2026, at 08:55, 10:05, 11:05, 12:05, 13:05, 14:05 and 15:05 New York time, shortly before each round deadline |
| `icaif-official-v391.service` | What runs and under which restrictions (see below) |
| `10-credentials-state.conf` | Drop-in: grants write access to the private state folder |
| `20-…`, `40-…`, `50-v3933-adapter.conf` | Drop-ins that switched the service to each new release folder; `50-` is the final, active one |
| `effective_*.txt` | Snapshot of the merged configuration systemd actually ran (main file plus all drop-ins) |
| `ORIGINAL_HASHES_AND_REDACTIONS.json` | Fingerprints of the original files and a list of redacted lines |

## Service settings explained

| Setting | Meaning | Why |
| --- | --- | --- |
| `Type=oneshot` | Runs once per start, then exits | Each round is a short, fresh run |
| `User=icaif` | Dedicated, unprivileged Linux account | Limits what the program can affect on the server |
| `UMask=0077` | New files are private to that account | Journal and credentials stay unreadable to others |
| `TimeoutStartSec=600` | Stopped after 10 minutes | A hung run cannot block the next round |
| `Restart=no` | Never retried automatically | An automatic retry could risk a second submission |
| `NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome`, `PrivateTmp` | Sandboxing | No admin rights, no writes to system or home folders |
| `ReadWritePaths=…` | The only writable folders | Market-data cache and the agent's own journal |

## How fixes were deployed

Each fix shipped as a new release folder plus a drop-in that pointed `ExecStart` at it. The timer and the main service file never changed. This made each change small, reviewable and easy to roll back.

For a template of how the refactored layout would be deployed, see [example/](example/README.md).