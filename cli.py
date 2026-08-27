"""Run the agent, or ask it what it is allowed to do.

    python cli.py --capabilities                     # the whole action table
    python cli.py --scenario memory_leak             # simulated, dry run
    python cli.py --scenario memory_leak --execute --auto-remediate
    python cli.py --scenario disk_pressure --execute --json
    python cli.py --real --ticks 20                  # this machine, read-only

Defaults are the cautious ones: simulated system, dry run, no auto-remediation.
Every dangerous thing needs a flag, and the destructive tier needs approval
that no flag here grants.

--json prints the incident records rather than the report, so another program
(or an agent) can drive this and read the result. That is the machine
interface: `--capabilities` says what can be asked for, `--json` says what
happened, and `safety.Policy` refuses anything outside the table.
"""

import argparse
import json
import sys

from agent import HealingAgent
from monitor import RealSystem, SimulatedSystem
from safety import Policy, describe_capabilities

SCENARIOS = {
    "memory_leak": ("inject_memory_leak", {}),
    "memory_spike": ("inject_memory_spike", {}),
    "cpu_spike": ("inject_cpu_spike", {}),
    "disk_pressure": ("inject_disk_pressure", {}),
    "healthy": (None, {}),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="memory_leak")
    ap.add_argument("--real", action="store_true",
                    help="watch THIS machine via psutil instead of the simulator")
    ap.add_argument("--ticks", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=14,
                    help="samples collected before any anomaly is reported")
    ap.add_argument("--execute", action="store_true",
                    help="actually run remediations (default: describe only)")
    ap.add_argument("--auto-remediate", action="store_true",
                    help="allow the disruptive tier without asking")
    ap.add_argument("--allow-service", action="append", default=None,
                    help="a service the agent may restart; repeatable")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", help="write the incident records to this JSON file")
    ap.add_argument("--capabilities", action="store_true",
                    help="print the action table and exit")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.capabilities:
        rows = describe_capabilities()
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print(f"{'action':<18}{'tier':<14}{'reversible':<12}description")
            for r in rows:
                print(f"{r['action']:<18}{r['tier']:<14}"
                      f"{'yes' if r['reversible'] else 'NO':<12}{r['description']}")
        return 0

    if args.real:
        # Refuse the combination rather than trusting a flag: the simulator is
        # where remediation gets exercised, and RealSystem.restart_service /
        # free_disk raise NotImplementedError by design.
        if args.execute:
            print("--real --execute is not supported: no init-system adapter is "
                  "implemented, so there is nothing to execute. Watch and "
                  "diagnose with --real; remediate in the simulator.",
                  file=sys.stderr)
            return 2
        system = RealSystem()
    else:
        system = SimulatedSystem(seed=args.seed)

    allowed = set(args.allow_service or ["app-worker", "nginx", "postgres"])
    agent = HealingAgent(system, Policy(
        dry_run=not args.execute, auto_remediate=args.auto_remediate,
        allowed_services=allowed))

    for _ in range(args.warmup):
        agent.observe()
        if hasattr(system, "advance"):
            system.advance()

    injector, kwargs = SCENARIOS[args.scenario]
    if injector and not args.real:
        getattr(system, injector)(**kwargs)

    for _ in range(args.ticks):
        if hasattr(system, "advance"):
            system.advance()
        agent.step()

    if args.report:
        agent.write_reports(args.report)

    summary = agent.summary()
    if args.json:
        print(json.dumps({
            "scenario": "real" if args.real else args.scenario,
            "dry_run": not args.execute,
            "ticks": args.ticks,
            "summary": summary,
            "incidents": [i.to_dict() for i in agent.incidents],
        }, indent=2, default=str))
        return 0

    if not agent.incidents:
        print(f"{args.ticks} passes, nothing anomalous. "
              f"(warmup {args.warmup} samples)")
        return 0
    for incident in agent.incidents:
        print(incident.report())
        print()
    print("  ".join(f"{k}: {v}" for k, v in summary.items() if k != "causes"))
    print("causes: " + ", ".join(f"{k}={v}" for k, v in summary["causes"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
