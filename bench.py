"""The agent against the thing it replaces: a static-threshold script.

    python bench.py
    python bench.py --json
    python bench.py --trials 100 --threshold 85

THE BASELINE HAS TO BE ABLE TO WIN
----------------------------------
The honest comparison is not "agent vs nothing". Every shop already has the
script: alert when memory > 90%, then restart the biggest process. It is three
lines, it needs no baseline, no warmup and no history, and on a machine that
genuinely runs out of memory it fires and it works. If the agent cannot beat
that on a number, the agent is ceremony.

So both run the same seeded scenarios and are scored on the same four things:

    acted               did it take a remediation at all
    mean_ticks_to_act   how long it took (5 simulated seconds per tick)
    false_positives     did it act on a scenario where acting is wrong
    verified_fixed      did the metric actually come back down afterwards

`false_positives` is the column that matters and the one a detection-rate-only
report hides. Two scenarios here are traps:

    healthy              nothing wrong. Any action is a false positive.
    memory_spike_stable  a process jumps to a large but STABLE footprint -- a
                         cache warming, a batch job loading a dataset.
                         Restarting it is a self-inflicted outage, and it is
                         indistinguishable from a leak on its first sample.

The threshold script cannot express "high but not climbing", so it fires or
misses purely on where the threshold sits. That is the whole argument, and
this file is where it becomes a number instead of a claim.
"""

import argparse
import json
import statistics

from agent import HealingAgent
from monitor import SimulatedSystem
from safety import Policy

SERVICES = {"app-worker", "nginx", "postgres", "metrics-agent"}
WARMUP_TICKS = 14          # the agent's baselines need history; the script does not
MAX_TICKS = 40

# (name, injector, kwargs, acting_is_correct)
SCENARIOS = [
    ("memory_leak", "inject_memory_leak", {}, True),
    ("cpu_spike", "inject_cpu_spike", {}, True),
    ("disk_pressure", "inject_disk_pressure", {}, True),
    ("memory_spike_stable", "inject_memory_spike", {}, False),
    ("healthy", None, {}, False),
]


def _policy():
    return Policy(dry_run=False, auto_remediate=True, allowed_services=set(SERVICES))


def run_agent(scenario, seed):
    """The full loop: baselines, trend, gate, act, verify."""
    _, injector, kwargs, _ = scenario
    system = SimulatedSystem(seed=seed)
    agent = HealingAgent(system, _policy())
    for _ in range(WARMUP_TICKS):
        agent.observe()
        system.advance()
    if injector:
        getattr(system, injector)(**kwargs)

    for tick in range(MAX_TICKS):
        system.advance()
        incident = agent.step()
        if incident and any(a.executed for a in incident.actions):
            return {"acted": True, "ticks": tick,
                    "verified": bool(incident.verified),
                    "cause": incident.diagnosis.cause}
    return {"acted": False, "ticks": None, "verified": False,
            "cause": agent.incidents[-1].diagnosis.cause if agent.incidents else None}


def run_threshold(scenario, seed, threshold):
    """The script. No history, no trend, no verify -- act on the first breach.

    Deliberately given the same simulated machine and the same remediations as
    the agent, so the only thing measured is *when it decides to act*, not what
    it is allowed to do.
    """
    _, injector, kwargs, _ = scenario
    system = SimulatedSystem(seed=seed)
    for _ in range(WARMUP_TICKS):
        system.sample()
        system.advance()
    if injector:
        getattr(system, injector)(**kwargs)

    for tick in range(MAX_TICKS):
        system.advance()
        sample = system.sample()

        if sample.disk_percent >= threshold:
            system.free_disk(20000)
            after = system.sample()
            return {"acted": True, "ticks": tick, "cause": "disk_pressure",
                    "verified": after.disk_percent < sample.disk_percent * 0.95}

        on_memory = sample.memory_percent >= threshold
        if on_memory or sample.cpu_percent >= threshold:
            worst = (sample.top_by_memory(1) if on_memory else sample.top_by_cpu(1))[0]
            if worst.name not in SERVICES:
                continue
            before = sample.memory_percent if on_memory else sample.cpu_percent
            system.restart_service(worst.name)
            for _ in range(3):
                system.advance()
            settled = system.sample()
            after = settled.memory_percent if on_memory else settled.cpu_percent
            return {"acted": True, "ticks": tick, "cause": "threshold_breach",
                    "verified": after < before * 0.95}

    return {"acted": False, "ticks": None, "verified": False, "cause": None}


def score(runs, acting_is_correct):
    acted = [r for r in runs if r["acted"]]
    ticks = [r["ticks"] for r in acted if r["ticks"] is not None]
    return {
        "acted": len(acted),
        "of": len(runs),
        "false_positives": 0 if acting_is_correct else len(acted),
        "missed": len(runs) - len(acted) if acting_is_correct else 0,
        "verified_fixed": sum(1 for r in acted if r["verified"]),
        "mean_ticks_to_act": round(statistics.mean(ticks), 2) if ticks else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=90.0,
                    help="the script's alert level, percent")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report = {"trials": args.trials, "threshold": args.threshold, "scenarios": {}}
    for scenario in SCENARIOS:
        name, _, _, correct = scenario
        seeds = range(1, args.trials + 1)
        report["scenarios"][name] = {
            "acting_is_correct": correct,
            "agent": score([run_agent(scenario, s) for s in seeds], correct),
            "threshold": score([run_threshold(scenario, s, args.threshold)
                                for s in seeds], correct),
        }

    totals = {}
    for who in ("agent", "threshold"):
        rows = [v[who] for v in report["scenarios"].values()]
        totals[who] = {
            "false_positives": sum(r["false_positives"] for r in rows),
            "missed": sum(r["missed"] for r in rows),
            "verified_fixed": sum(r["verified_fixed"] for r in rows),
        }
    report["totals"] = totals

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"{args.trials} seeded trials per scenario, "
          f"script threshold {args.threshold:.0f}%\n")
    print(f"{'':<24}{'acted':>8}{'missed':>8}{'false+':>8}{'verified':>10}{'ticks':>8}")
    for name, row in report["scenarios"].items():
        print(f"{name}  ({'act' if row['acting_is_correct'] else 'DO NOT act'})")
        for who in ("agent", "threshold"):
            r = row[who]
            ticks = "-" if r["mean_ticks_to_act"] is None else f"{r['mean_ticks_to_act']:.1f}"
            print(f"  {who:<22}{r['acted']:>8}{r['missed']:>8}"
                  f"{r['false_positives']:>8}{r['verified_fixed']:>10}{ticks:>8}")
    print()
    for who, t in totals.items():
        print(f"{who:<12} false positives {t['false_positives']:>3}   "
              f"missed {t['missed']:>3}   verified fixes {t['verified_fixed']:>3}")


if __name__ == "__main__":
    main()
