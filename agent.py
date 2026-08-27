"""The observe -> diagnose -> act -> verify loop, and the incident record.

THE LOOP
--------
    observe    sample metrics, update baselines
    diagnose   turn anomalies into a named root cause with evidence
    propose    map the diagnosis to a remediation from the capability table
    gate       safety.Policy decides; a refusal is a normal outcome
    act        execute, or record the proposal for a human
    verify     sample again -- did it actually work?

**Verify is the step that separates this from a script.** A script restarts
the service and reports success because the restart command returned zero. An
agent checks whether the metric that triggered the incident actually came
back down, and says so when it did not. Without it, "self-healing" means
"performed an action", which is not the same thing and is much easier.

DIAGNOSIS IS RULES, AND SAYS SO WHEN IT DOES NOT KNOW
-----------------------------------------------------
`diagnose()` is a small set of rules over the metrics, ordered by how
unambiguous the cause is. Every branch can return confidence `"unsure"`, which
makes the diagnosis non-actionable and stops the loop before `propose()` --
the agent watches another pass instead of guessing.

There is no model in the diagnosis path. `bench.py` measures this loop against
the thing it actually replaces -- a static-threshold script -- because that is
the comparison with a number on both sides. A model was not added on top of a
baseline that could not be scored without a credential; see RESULTS.md.
"""

import json
import time
from dataclasses import asdict, dataclass, field

from monitor import Monitor
from safety import CAPABILITIES, Decision, Policy

# A leak rises steadily; a spike is high and flat. Megabytes per sample, on the
# leaking process's own series -- sampling noise is a few MB, a real leak is
# tens to hundreds, so anything in between is deliberately not called a leak.
LEAK_SLOPE_MB = 20.0
# Enough samples for a slope to mean anything.
TREND_SAMPLES = 8


@dataclass
class Diagnosis:
    cause: str                 # memory_leak | runaway_process | disk_pressure | ...
    confidence: str            # certain | likely | unsure
    summary: str
    evidence: dict = field(default_factory=dict)
    source: str = "rules"

    @property
    def actionable(self):
        return self.cause != "unknown" and self.confidence != "unsure"


@dataclass
class ActionRecord:
    action: str
    target: str
    tier: str
    executed: bool
    decision: str
    result: str = ""


@dataclass
class Incident:
    id: str
    opened_at: float
    trigger: str
    diagnosis: Diagnosis = None
    actions: list = field(default_factory=list)
    verified: bool = None
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    closed_at: float = None
    outcome: str = "open"      # open | resolved | unresolved | awaiting-approval
                               # | observing
    repeats: int = 0           # further passes that saw the same condition

    def to_dict(self):
        out = asdict(self)
        out["diagnosis"] = asdict(self.diagnosis) if self.diagnosis else None
        return out

    def report(self):
        """A human-readable incident report.

        Written to be read by whoever is on call at 3am, which means the
        first three lines answer: what broke, what was done, did it work.
        """
        lines = [
            f"INCIDENT {self.id}",
            f"  trigger    {self.trigger}",
            f"  cause      {self.diagnosis.summary if self.diagnosis else 'not diagnosed'}",
            f"  outcome    {self.outcome}"
            + (f" ({self.repeats} further passes)" if self.repeats else "")
            + ("" if self.verified is None else
               f" (verified: {'yes' if self.verified else 'NO'})"),
        ]
        if self.diagnosis:
            lines.append(f"  confidence {self.diagnosis.confidence} "
                         f"(from {self.diagnosis.source})")
        if self.actions:
            lines.append("  actions")
            for a in self.actions:
                mark = "done" if a.executed else "blocked"
                lines.append(f"    [{mark:>7}] {a.action} {a.target}  -- {a.decision}")
                if a.result:
                    lines.append(f"              {a.result}")
        else:
            lines.append("  actions    none taken")

        if self.before and self.after:
            lines.append("  metrics    " + "  ".join(
                f"{k}: {self.before[k]:.1f} -> {self.after[k]:.1f}"
                for k in sorted(self.before) if k in self.after))
        return "\n".join(lines)


class HealingAgent:
    """Watches a system and tries to fix what goes wrong with it."""

    def __init__(self, system, policy=None, clock=time.time, window=60):
        self.system = system
        self.policy = policy or Policy()
        self.clock = clock
        self.monitor = Monitor(system, window=window)
        self.incidents = []
        self._next_id = 1

    # -- the loop --------------------------------------------------------

    def observe(self):
        return self.monitor.collect()

    def step(self):
        """One full pass. Returns an Incident if something happened."""
        sample = self.observe()
        anomalies = self.monitor.anomalies(sample)
        if not anomalies:
            return None

        # One incident per pass, for the worst anomaly. Opening five incidents
        # for five symptoms of one leak is how an on-call queue becomes
        # unreadable -- and they would all propose the same fix.
        worst = anomalies[0]
        diagnosis = self._diagnose_rules(sample, anomalies)

        # The same rule across time, not just across symptoms. A condition the
        # agent is deliberately watching stays ONE incident with a repeat
        # count; a new record every five seconds is the same unreadable queue
        # arriving more slowly.
        last = self.incidents[-1] if self.incidents else None
        if (last is not None and last.outcome == "observing"
                and last.diagnosis and last.diagnosis.cause == diagnosis.cause):
            last.repeats += 1
            last.diagnosis = diagnosis      # keep the freshest evidence
            last.closed_at = self.clock()
            return None
        incident = Incident(
            id=f"INC-{self._next_id:04d}", opened_at=self.clock(),
            trigger=worst.detail,
            before={"cpu": sample.cpu_percent, "memory": sample.memory_percent,
                    "disk": sample.disk_percent})
        self._next_id += 1

        incident.diagnosis = diagnosis
        proposals = self.propose(diagnosis, sample)

        for action, target, note in proposals[:self.policy.max_actions_per_incident]:
            record = self._attempt(action, target, note)
            incident.actions.append(record)
            if record.executed:
                break        # act once, then verify; never chain blind

        incident.after, incident.verified = self.verify(incident)
        incident.closed_at = self.clock()
        incident.outcome = (
            # Deciding not to act is a distinct outcome from acting and
            # failing. Collapsing them makes a cautious agent look broken --
            # and "no remediation exists for this cause" (memory_spike) is a
            # deliberate design decision, not an unresolved incident.
            "observing" if not incident.actions else
            "resolved" if incident.verified else
            "awaiting-approval" if any(not a.executed and "approval" in a.decision
                                       for a in incident.actions) else
            "unresolved")

        self.incidents.append(incident)
        return incident

    # -- diagnosis -------------------------------------------------------

    def _diagnose_rules(self, sample, anomalies):
        by_metric = {a.metric: a for a in anomalies}

        # Disk first: it is the one that takes the machine down hardest and
        # the one with the least ambiguous cause.
        disk = by_metric.get("disk_percent")
        if disk:
            slope = self.monitor.trend("disk_percent", TREND_SAMPLES)
            return Diagnosis(
                "disk_pressure",
                "certain" if disk.severity == "critical" else "likely",
                f"disk at {disk.value:.1f}% and "
                + (f"rising {slope:.2f}%/sample" if slope and slope > 0 else "not falling"),
                {"disk_percent": disk.value, "free_mb": round(sample.disk_free_mb),
                 "slope": round(slope, 4) if slope is not None else None})

        # A process whose memory is climbing steadily is a leak. One that is
        # simply large is not -- and restarting a large-but-stable service is
        # an outage caused by a false positive.
        #
        # The slope comes from the process's OWN series over the run it has
        # been abnormal, not from system memory over a fixed window. See
        # Monitor.process_trend for the bug that distinction fixes.
        process_memory = [a for a in anomalies if a.metric.endswith(":memory")]
        if process_memory:
            worst = process_memory[0]
            name = worst.evidence.get("name", "?")
            evidence = {"process": name, "pid": worst.evidence.get("pid"),
                        "memory_mb": worst.value, "baseline_mb": worst.baseline}
            slope, run = self.monitor.process_trend(name, "memory", TREND_SAMPLES)

            if slope is None:
                # Genuinely undecidable, not a gap in the rules. On one sample
                # a leak and a spike are the same observation. Confidence
                # "unsure" makes this non-actionable, so the agent watches
                # another tick instead of restarting a service on a guess.
                return Diagnosis(
                    "memory_growth", "unsure",
                    f"{name} at {worst.value:.0f}MB against a usual "
                    f"{worst.baseline:.0f}MB, abnormal for {run} sample(s) -- "
                    f"too few to tell a leak from a spike; observing",
                    dict(evidence, abnormal_samples=run))

            if slope > LEAK_SLOPE_MB:
                return Diagnosis(
                    "memory_leak", "likely",
                    f"{name} memory climbing steadily "
                    f"({worst.baseline:.0f}MB -> {worst.value:.0f}MB, "
                    f"+{slope:.0f}MB/sample over {run} samples)",
                    dict(evidence, slope_mb=round(slope, 2), abnormal_samples=run))

            return Diagnosis(
                "memory_spike", "likely",
                f"{name} using {worst.value:.0f}MB against a usual "
                f"{worst.baseline:.0f}MB, flat for {run} samples "
                f"({slope:+.1f}MB/sample) -- high, not climbing",
                dict(evidence, slope_mb=round(slope, 2), abnormal_samples=run))

        process_cpu = [a for a in anomalies if a.metric.endswith(":cpu")]
        if process_cpu:
            worst = process_cpu[0]
            return Diagnosis(
                "runaway_process", "likely",
                f"{worst.evidence.get('name', '?')} at {worst.value:.0f}% CPU "
                f"against a usual {worst.baseline:.0f}%",
                {"process": worst.evidence.get("name"),
                 "pid": worst.evidence.get("pid"), "cpu_percent": worst.value})

        memory = by_metric.get("memory_percent")
        if memory:
            return Diagnosis(
                "memory_pressure", "likely",
                f"system memory at {memory.value:.1f}% with no single process "
                f"responsible",
                {"memory_percent": memory.value,
                 "top": [(p.name, round(p.memory_mb)) for p in sample.top_by_memory(3)]})

        cpu = by_metric.get("cpu_percent")
        if cpu:
            return Diagnosis(
                "cpu_pressure", "likely",
                f"system CPU at {cpu.value:.1f}% with no single process responsible",
                {"cpu_percent": cpu.value,
                 "top": [(p.name, round(p.cpu_percent)) for p in sample.top_by_cpu(3)]})

        # Anomalies exist but no rule names them. Saying so is the correct
        # output; guessing a cause here is how an agent takes a confident
        # wrong action.
        return Diagnosis(
            "unknown", "unsure",
            f"{len(anomalies)} anomalies with no matching rule",
            {"anomalies": [a.detail for a in anomalies[:4]]})

    # -- remediation -----------------------------------------------------

    def propose(self, diagnosis, sample):
        """Map a cause to ordered remediation attempts.

        Ordered least-drastic first, and the loop stops at the first one that
        executes. A reversible fix that might work is always tried before an
        irreversible one that definitely would.
        """
        if diagnosis is None or not diagnosis.actionable:
            return []

        evidence = diagnosis.evidence
        name = evidence.get("process")
        pid = evidence.get("pid")

        if diagnosis.cause == "disk_pressure":
            return [("rotate_logs", "/var/log", "compress and roll logs first"),
                    ("clear_cache", "/var/cache", "regenerable, so safe to drop"),
                    ("delete_files", "/var/tmp", "irreversible; needs approval")]

        if diagnosis.cause in ("memory_leak", "runaway_process"):
            proposals = []
            if name:
                proposals.append(
                    ("restart_service", name,
                     "reversible, and a restart clears a leak without losing the service"))
            if pid:
                proposals.append(
                    ("kill_process", str(pid),
                     "irreversible; only if a restart is not available"))
            return proposals

        if diagnosis.cause == "memory_spike":
            # Deliberately no remediation. A large-but-stable process is
            # usually doing its job. Restarting it is an outage caused by a
            # false positive, so the agent reports and stops.
            return []

        if diagnosis.cause in ("memory_pressure", "cpu_pressure"):
            return []

        return []

    def _attempt(self, action, target, note):
        decision = self.policy.check(action, target)
        record = ActionRecord(action=action, target=str(target),
                              tier=decision.tier, executed=False,
                              decision=decision.reason)
        if not decision.allowed:
            return record

        if self.policy.dry_run:
            record.decision = f"permitted, but dry-run is on ({note})"
            return record

        try:
            record.result = self._execute(action, target)
            record.executed = True
        except Exception as exc:
            record.result = f"failed: {type(exc).__name__}: {exc}"
        return record

    def _execute(self, action, target):
        """The only place the agent touches the system.

        Everything routes through here so there is exactly one code path to
        audit, and `safety.Policy.check` has already run on all of it.
        """
        if action == "kill_process":
            self.system.kill_process(int(target))
            return f"terminated pid {target}"
        if action == "restart_service":
            self.system.restart_service(target)
            return f"restarted {target}"
        if action in ("rotate_logs", "clear_cache", "delete_files"):
            freed = self.system.free_disk(20000)
            return f"freed {freed:.0f}MB from {target}"
        if action == "throttle_process":
            return f"lowered priority of {target}"
        raise NotImplementedError(f"no executor for {action}")

    # -- verification ----------------------------------------------------

    def verify(self, incident, settle_samples=3):
        """Did the metric that triggered this actually come back down?

        THE step that separates an agent from a script. A script reports
        success because the command exited zero. This re-measures.
        """
        for _ in range(settle_samples):
            if hasattr(self.system, "advance"):
                self.system.advance()
            sample = self.observe()

        after = {"cpu": sample.cpu_percent, "memory": sample.memory_percent,
                 "disk": sample.disk_percent}

        if not any(a.executed for a in incident.actions):
            return after, None          # nothing was done; nothing to verify

        # Compare only the metric the incident was opened on. A leak fix that
        # also happened to coincide with a CPU dip is not evidence.
        key = ("disk" if "disk" in incident.trigger else
               "memory" if "memory" in incident.trigger.lower() or "MB" in incident.trigger
               else "cpu")
        before_value = incident.before.get(key)
        after_value = after.get(key)
        if before_value is None or after_value is None:
            return after, None
        return after, after_value < before_value * 0.95

    # -- reporting -------------------------------------------------------

    def summary(self):
        resolved = [i for i in self.incidents if i.outcome == "resolved"]
        blocked = [i for i in self.incidents if i.outcome == "awaiting-approval"]
        return {
            "incidents": len(self.incidents),
            "resolved": len(resolved),
            "unresolved": len([i for i in self.incidents if i.outcome == "unresolved"]),
            "observing": len([i for i in self.incidents if i.outcome == "observing"]),
            "awaiting_approval": len(blocked),
            "actions_executed": sum(1 for i in self.incidents for a in i.actions
                                    if a.executed),
            "actions_blocked": sum(1 for i in self.incidents for a in i.actions
                                   if not a.executed),
            "causes": _count(i.diagnosis.cause for i in self.incidents if i.diagnosis),
        }

    def write_reports(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([i.to_dict() for i in self.incidents], fh, indent=2, default=str)
        return path


def _count(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
