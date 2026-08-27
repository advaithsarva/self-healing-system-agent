# Self-Healing System Agent

An agent that watches a machine, diagnoses what is going wrong, fixes it if it
is allowed to, and then **re-measures to check whether the fix worked**.

Rolling baselines instead of fixed thresholds, a static capability table
instead of free-form actions, and an incident record for every pass. Standard
library only; `psutil` is optional and only needed to watch a real machine.

**Across 200 seeded trials against a static-threshold script: 0 false
positives, 0 missed faults, 120/120 remediations verified — and it catches a
memory leak at 43% memory usage, 32 ticks before a 75% threshold script
notices.** Full numbers and the cases where the script wins in
[RESULTS.md](RESULTS.md).

```bash
python cli.py --scenario memory_leak --execute --auto-remediate
python cli.py --capabilities        # everything it could possibly do
python test_agent.py                # 21/21
python bench.py                     # agent vs the threshold script
```

---

## The rule the whole thing turns on

> **No action reaches the system unless it is on the allowlist, and every
> destructive action is either reversible or blocked pending human approval.**

That is the first line of `safety.py`, which is the first file in the project
rather than the last. The premise — an autonomous process that kills processes
and deletes files — is only defensible if the answer to *"what could it
possibly do?"* is a finite list that fits on a screen.

The failure being prevented is not a crash. It is an agent that reasons its way,
one plausible step at a time, into `kill -9` on the database because the
database was using the most memory. Every individual step looks sound. The
outcome is an outage caused by the thing that was supposed to prevent one.

So capability is **not** derived from reasoning. It is a static table, checked
before execution, and the diagnosis layer can only select from it:

| tier | reversible? | runs automatically? |
|---|---|---|
| `safe` — rotate logs, clear cache, read anything | yes | always |
| `disruptive` — restart a service, throttle a process | yes, users notice | only with `--auto-remediate` |
| `destructive` — kill a process, delete files, patch config | **no** | **never.** Human approval, always. No flag disables this. |

On top of that, a harder rule: `systemd`, `sshd`, `lsass.exe` and the agent's
own PID are untouchable at every tier. Losing SSH means losing the ability to
fix whatever the agent just did.

---

## Verify is the step that makes it an agent

A script restarts the service and reports success because the restart command
returned zero. This re-samples the metric that opened the incident and says
whether it actually came back down.

```
observe    sample metrics, update baselines
diagnose   turn anomalies into a named cause -- or say "unsure"
propose    map the cause to remediations, least drastic first
gate       safety.Policy decides; a refusal is a normal, recorded outcome
act        execute exactly one, then stop
verify     sample again. Did the number move?
```

Without the last step, "self-healing" means "performed an action", which is a
much easier thing to build and a much weaker thing to claim.

---

## Baselines, not thresholds

`Alert if memory > 90%` is wrong in both directions. A build server sitting at
92% is fine. A database that normally runs at 40% and is now at 70% has a
problem worth catching two hours before it hits 90.

So every metric keeps a rolling median and a MAD-based deviation, and anomalies
are z-scores against that — with an absolute ceiling kept only as a backstop
for the genuinely urgent. Two details matter more than they look:

- **The deviation is a median absolute deviation, not a standard deviation.**
  Once a metric goes anomalous it starts inflating its own baseline's variance,
  which raises the bar and hides the anomaly being tracked. That is a detector
  suppressing itself, exactly when it matters.
- **The deviation has a floor, in the metric's own units.** A z-score is a
  ratio, and a ratio with a near-zero denominator is not a detector — see the
  false positive below.

The cost is a warmup during which the agent knows nothing and says so, rather
than firing on the first sample it has ever seen.

---

## Three bugs worth keeping

**1. A memory leak was diagnosed as a memory spike.** The anomaly fires on the
*first* abnormal sample, and the rule then asked for the slope of **system**
memory over the last 8 samples — 7 of which predated the leak entirely. The
slope described the onset step, not the trend, so a process climbing at
180 MB/tick was reported as "high, but not climbing" — a cause with **no
remediation attached**, so the agent watched the machine fill up and did
nothing.

Two things fix it, and both are needed: read **the leaking process's own
series**, and fit the slope over **only the samples since it went abnormal**.
`Monitor.process_trend` does both.

**2. A z-score with nothing in the denominator.** `sshd` sits at 0.1% CPU and
never moves, so its MAD collapses to roughly zero — and ordinary jitter taking
it to 1.0% CPU scored **3.1 deviations** and opened an incident proposing a
restart of `sshd`, on a machine with nothing wrong with it. Three healthy runs
out of ten did this.

The fix is not a bigger `Z_THRESHOLD` — that trades these false positives for
missed real ones on noisier metrics. It is `min_scale()`: name the change too
small to be worth calling an anomaly *in the metric's own units* (one
percentage point of CPU, fifteen megabytes of RSS) and floor the denominator
there. Below that, no multiple of it is a finding.

**3. "Unsure" had to become a real answer.** On its first abnormal sample a
leak and a spike are *the same observation*. There is no cleverness that
separates them, and restarting a service on one sample is precisely the false
positive that gets an agent switched off. So the diagnosis returns confidence
`unsure`, which makes it non-actionable, and the agent watches another pass.
It costs about five seconds and it is the difference between the two rows in
the `memory_spike_stable` block of [RESULTS.md](RESULTS.md).

---

## Why the machine is simulated

Testing an agent that fixes broken machines means either waiting for a real
memory leak, or deliberately causing one on the machine running the tests — so
in practice nobody tests it, and the detection thresholds are guesses that were
never checked against a case they were supposed to catch.

`SimulatedSystem` produces a leak, a runaway process, disk pressure or a
large-but-stable process on demand, deterministically, behind the same
interface as `RealSystem`. The agent cannot tell them apart. That is what makes
`bench.py` possible at all: 200 trials with known ground truth, including the
two scenarios where the correct behaviour is **to do nothing**.

`RealSystem` (psutil) is read-only. Its `restart_service` and `free_disk` raise
`NotImplementedError` rather than guessing at an init system, and `cli.py`
refuses `--real --execute` for that reason. **The remediation path has never
been run against a real machine, and RESULTS.md says so.**

---

## Files

| file | what is in it |
|---|---|
| `safety.py` | the capability table, the tiers, the protected lists, the single gate |
| `monitor.py` | sampling, rolling baselines, anomaly detection, the simulator |
| `agent.py` | the observe/diagnose/propose/gate/act/verify loop and the incident record |
| `cli.py` | run it; `--capabilities`, `--json` |
| `bench.py` | the agent against a static-threshold script, 5 scenarios |
| `test_agent.py` | 21 tests, one per real bug, plus a check that they fail on the old code |

```bash
pip install psutil     # only for --real; everything else is stdlib
```
