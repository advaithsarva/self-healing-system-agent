# Results

Every number below has the command that produced it printed above it. All runs
are seeded; the simulator is deterministic, so they reproduce exactly.

Environment: Windows 11, Python 3.12.1, standard library only (`psutil` used
for the `--real` section).

---

## 1. The agent against a static-threshold script

The honest baseline is not "nothing". It is the script every shop already has:
alert when a metric crosses 90%, then restart the biggest process. Three lines,
no baseline, no warmup, no history. Both are given the **same simulated
machine and the same remediations**, so the only thing being measured is *when
each one decides to act*.

Five scenarios, and two of them are traps where the correct behaviour is to do
nothing:

| scenario | correct behaviour |
|---|---|
| `memory_leak` | act — `app-worker` climbing 180 MB/tick |
| `cpu_spike` | act — `app-worker` pinned at 340% CPU |
| `disk_pressure` | act — disk filling 6 GB/tick |
| `memory_spike_stable` | **do not act** — `postgres` jumps +1400 MB and stays flat |
| `healthy` | **do not act** — nothing is wrong |

`memory_spike_stable` is the case that matters. A cache warming up or a batch
job loading a dataset is indistinguishable from the first sample of a leak.
Restarting it is a self-inflicted outage.

```bash
python bench.py --trials 40 --threshold 90
```

```
                           acted  missed  false+  verified   ticks
memory_leak  (act)
  agent                       40       0       0        40     1.0
  threshold                    0      40       0         0       -
cpu_spike  (act)
  agent                       40       0       0        40     0.0
  threshold                    7      33       0         7    19.4
disk_pressure  (act)
  agent                       40       0       0        40     2.0
  threshold                   40       0       0         0    29.0
memory_spike_stable  (DO NOT act)
  agent                        0       0       0         0       -
  threshold                    0       0       0         0       -
healthy  (DO NOT act)
  agent                        0       0       0         0       -
  threshold                    0       0       0         0       -

agent        false positives   0   missed   0   verified fixes 120
threshold    false positives   0   missed  73   verified fixes   7
```

**200 trials: the agent gets 0 false positives, 0 missed faults, and all 120
remediations verified. The 90% script misses 73 of 120.**

`ticks` is passes-to-action, 5 simulated seconds each. The agent acts on the
memory leak on tick 1 — one pass after `unsure` becomes decidable.

### The script tuned until it can win

A 90% threshold losing is not an interesting result on its own; the threshold
is a knob. So it was turned down until the script matched:

```bash
python bench.py --trials 40 --threshold 85
python bench.py --trials 40 --threshold 75
```

| script threshold | missed / 120 | verified fixes | ticks to catch the leak |
|---|---|---|---|
| 90% | 73 | 7 | never |
| 85% | 40 | 40 | never |
| **75%** | **0** | **120** | **33.0** |
| *agent* | *0* | *120* | *1.0* |

**At 75% the script matches the agent on every count column.** That is the
honest version of this comparison, and it is worth more than the 90% table: a
threshold script is not incapable, it is *late*. Catching the leak takes it 33
ticks against the agent's 1 — by which point the leaking process has grown from
700 MB to roughly 6.6 GB and system memory has gone from 39% to 75%.

The agent's advantage is stated precisely: **32 ticks (~160 simulated seconds)
of earlier warning on the leak, and 15 ticks on disk pressure, at equal
accuracy.** Not "it detects things the script cannot".

Two caveats, because both cut against the headline:

- **Neither approach false-positives on `memory_spike_stable` here**, but for
  different reasons. The agent sees it, diagnoses `memory_spike`, and
  *declines to act* — visible as an `observing` incident in the record. The
  script never sees it at all: +1400 MB on this simulated machine only reaches
  47% system memory, below any threshold that is not already firing constantly.
  The script's clean column is blindness, not judgement, and this benchmark
  cannot tell those apart. A per-process threshold rule would fail this case;
  a system-wide one is simply not looking.
- **The `verified` column is harsh on the script for a mechanical reason.**
  Verification asks whether the metric dropped below 95% of its pre-action
  value. Freeing a fixed 20 GB from a 500 GB disk is 3.9 percentage points, and
  3.9 points is a large enough *relative* drop at 56% (the agent's trigger
  point) but not at 90% (the script's). The same remedy verifies or does not
  depending on how late it was applied. That is a real property — acting late
  makes a fixed-size fix proportionally weaker — but it is not independent
  evidence, and the 75% row shows it flipping.

---

## 2. The bug this project spent the longest on

A memory leak was diagnosed as `memory_spike` — "high, but not climbing" — a
cause with **no remediation attached**, so the agent watched the machine fill
up and did nothing.

The anomaly fires on the **first** abnormal sample. The old rule then asked for
the slope of *system* memory over the last 8 samples, of which 7 predated the
leak entirely, so the slope described the onset step rather than the trend.

Both halves of the fix are load-bearing, and the second is the non-obvious one:

```bash
python test_agent.py     # "the leak slope comes from the process's own series"
```

| slope source, 6 samples into a 180 MB/tick leak | value |
|---|---|
| system memory, fixed 8-sample window (`Monitor.trend`) | **1.1 %/sample** |
| the leaking process's own series, abnormal run only (`Monitor.process_trend`) | **182 MB/sample** |

System memory is the sum over every process plus a 4 GB base, so one process
climbing is a small fraction of it and every other process moving is noise on
the signal being measured.

Fewer than 4 abnormal samples returns `(None, run_length)` and the diagnosis
becomes confidence `unsure` — non-actionable, so the loop stops before
`propose()`. **This is the correct answer, not a workaround.** On one sample a
leak and a spike are the same observation.

---

## 3. The false positive that a detection rate would have hidden

Before `min_scale()` existed, **3 of 10 healthy runs produced a remediation
proposal** — against `sshd` and `systemd`:

```
seed  2  tick 34  runaway_process | sshd (pid 410) using 1% of CPU,
                                    3.1 deviations above its usual 0%
seed 10  tick  0  runaway_process | systemd (pid 1) using 1% of CPU,
                                    3.3 deviations above its usual -0%
```

`sshd` sits at 0.1% CPU and never moves, so its MAD collapses to roughly zero
and the deviation was floored at `1e-6`. Ordinary gaussian jitter then scored
3.1 deviations. **A z-score is a ratio, and a ratio with a near-zero
denominator is not a detector.**

The fix is not a bigger `Z_THRESHOLD` — that trades these false positives for
missed real ones on noisier metrics. `min_scale()` names the change too small
to be worth calling an anomaly *in the metric's own units* and floors the
denominator there: 1 percentage point of CPU, 15 MB of RSS.

```bash
python test_agent.py     # "a healthy machine produces no action at all" (10 seeds)
```

After the fix: **0 of 10, and 0 of 40 in `bench.py`.**

Worth noting what saved this from being an outage rather than a bug report:
`safety.Policy` refused both actions anyway — `systemd` and `sshd` are on the
protected list and `kill_process` needs approval regardless. The gate is the
second line of defence and it held. The detector was still wrong.

---

## 4. Detection is early, and the ceiling still exists

```bash
python cli.py --scenario memory_leak --execute --auto-remediate
```

```
INCIDENT INC-0001
  trigger    app-worker (pid 1500) using 871MB of memory, 11.4 deviations above its usual 700MB
  cause      app-worker at 871MB against a usual 700MB, abnormal for 1 sample(s) --
             too few to tell a leak from a spike; observing
  outcome    observing

INCIDENT INC-0002
  trigger    app-worker (pid 1500) using 1604MB of memory, 60.2 deviations above its usual 701MB
  cause      app-worker memory climbing steadily (701MB -> 1604MB, +182MB/sample over 5 samples)
  outcome    resolved (verified: yes)
  actions
    [   done] restart_service app-worker  -- permitted by policy
  metrics    cpu: 4.1 -> 4.5  disk: 55.0 -> 55.0  memory: 43.9 -> 38.3
```

**The leak is remediated at 43.9% system memory** — a threshold script would
still be 30+ percentage points away from noticing. The pinned test is `a leak
is caught well below any 90% threshold`, which asserts `< 60%`.

The absolute ceiling (`MEMORY_CRITICAL`, `DISK_CRITICAL` = 95%) is kept as a
backstop, and it is not redundant: a baseline learned *during* a slow fill
calls the fill normal. Pinned by `the absolute ceiling fires even when the
baseline learned the fault`.

---

## 5. Declining to act is recorded, not hidden

```bash
python cli.py --scenario memory_spike --execute --auto-remediate
```

```
INCIDENT INC-0002
  trigger    postgres (pid 900) using 2803MB of memory, 93.4 deviations above its usual 1403MB
  cause      postgres using 2803MB against a usual 1414MB, flat for 13 samples
             (+0.1MB/sample) -- high, not climbing
  outcome    observing (5 further passes)
  actions    none taken

incidents: 2  resolved: 0  unresolved: 0  observing: 2  awaiting_approval: 0
actions_executed: 0  actions_blocked: 0
```

`observing` is a distinct outcome from `unresolved`. Collapsing them makes a
cautious agent look broken, and it is the difference between "chose not to act"
and "tried and failed".

The `(5 further passes)` counter is the same de-duplication rule applied across
time: a condition the agent is deliberately watching stays **one** incident. A
new record every five seconds is the same unreadable on-call queue arriving
more slowly.

---

## 6. Tests

```bash
python test_agent.py
```

```
21/21 passed
against the pre-fix diagnosis rule: 3/5 of the leak tests fail, as they must
```

The second line is the point of the suite. `verify_suite_is_not_decorative()`
re-installs the *previous* leak-vs-spike rule and re-runs the leak tests; 3 of
5 fail. A suite that passes on the broken code it was written for proves
nothing.

The two that still pass on the old code are honest about the bug's real
severity: with the old rule the agent misdiagnosed the **first** incident and
then self-corrected a tick later, once system memory had climbed enough for the
crude slope to clear its threshold. The failure was a confident wrong answer at
the moment of detection, not permanent paralysis — and the test that catches it
asserts exactly that: `all(i.diagnosis.cause != "memory_spike" for i in opened)`.

---

## 7. What has NOT been verified

Stated plainly, because the rest of this file is measured and this part is not.

- **No remediation has ever run against a real machine.**
  `RealSystem.restart_service` and `RealSystem.free_disk` raise
  `NotImplementedError` rather than guessing at systemd / `sc.exe` / launchctl,
  and `cli.py` refuses `--real --execute` outright. Every `verified: yes` in
  this document is against `SimulatedSystem`.
- **`RealSystem.kill_process` is implemented and has never been called.** It
  goes through the same `safety.Policy` gate, which is tested — the gate is
  verified, the call is not.
- The **read-only** real path has been exercised:

  ```bash
  python cli.py --real --ticks 5
  ```

  On this machine (busy, 80% memory) it opened 3 incidents, diagnosed
  `memory_growth` twice and `memory_spike` once, and **took no action**, which
  is correct for a machine that is loaded rather than failing. That validates
  sampling and diagnosis against real psutil data. It validates nothing about
  remediation.
- The simulator's fault dynamics are *chosen*, not measured from real
  incidents. A real leak is rarely a clean 180 MB per five seconds. The
  detection mechanism is what generalises; the specific tick counts do not.
