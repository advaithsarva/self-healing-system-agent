"""One test per bug that was actually found, plus the invariant.

    python test_agent.py

No pytest, no network, no real machine. Runs in about a second, so it gets run.

The last section, `test_old_diagnosis_fails_these_tests`, re-installs the
*previous* leak-vs-spike rule and asserts the suite catches it. A test suite
that passes on the broken code it was written for is decorative.
"""

import io
import os
import sys
from contextlib import redirect_stdout

import agent as agent_mod
import monitor as monitor_mod
from agent import HealingAgent, Diagnosis
from monitor import (MIN_TREND_RUN, Monitor, SimulatedSystem, WARMUP_SAMPLES,
                     min_scale)
from safety import CAPABILITIES, DESTRUCTIVE, Policy, describe_capabilities

SERVICES = {"app-worker", "nginx", "postgres", "metrics-agent"}

_results = []


def check(name):
    def wrap(fn):
        _results.append((name, fn))
        return fn
    return wrap


def build(seed=42, warmup=14, **policy_kw):
    """A warmed-up agent on a healthy simulated machine."""
    kw = dict(dry_run=False, auto_remediate=True, allowed_services=set(SERVICES))
    kw.update(policy_kw)
    system = SimulatedSystem(seed=seed)
    agent = HealingAgent(system, Policy(**kw))
    for _ in range(warmup):
        agent.observe()
        system.advance()
    return system, agent


def run_until(agent, system, ticks=25, stop=lambda inc: bool(inc.actions)):
    """Step the loop, returning every incident opened and the one that stopped it."""
    opened = []
    for _ in range(ticks):
        system.advance()
        incident = agent.step()
        if incident:
            opened.append(incident)
            if stop(incident):
                return opened, incident
    return opened, None


# --------------------------------------------------------------------------
# THE bug: a memory leak was diagnosed as a memory spike
# --------------------------------------------------------------------------

@check("a memory leak is diagnosed as a leak, not a spike")
def _():
    system, agent = build()
    system.inject_memory_leak()
    opened, acted = run_until(agent, system)
    assert acted is not None, "leak never produced a remediation"
    assert acted.diagnosis.cause == "memory_leak", f"diagnosed {acted.diagnosis.cause!r}"
    # The precise defect: the old rule read SYSTEM memory over a fixed window
    # that mostly predated the leak, so on the first abnormal sample it
    # confidently reported "high, but not climbing" -- a cause with no
    # remediation attached. Never say that about a live leak. "Unsure" is an
    # acceptable answer here; a confident wrong one is not.
    assert all(i.diagnosis.cause != "memory_spike" for i in opened), \
        "called a live leak 'high, not climbing'"
    assert "app-worker" in system.restarted


@check("the leak slope comes from the process's own series")
def _():
    # The distinction the bug turned on. System memory is the sum over every
    # process plus a 4GB base, so one process climbing is a small fraction of
    # it; the process's own series is not.
    system, agent = build()
    system.inject_memory_leak(mb_per_tick=180.0)
    for _ in range(MIN_TREND_RUN + 2):
        system.advance()
        agent.observe()
    slope, run = agent.monitor.process_trend("app-worker", "memory")
    system_slope = agent.monitor.trend("memory_percent", 8)
    assert slope is not None and slope > 100, f"process slope {slope}"
    assert run >= MIN_TREND_RUN
    # ~1.1 percentage points per sample: real, but nothing like 180.
    assert system_slope < 2.0, f"system slope {system_slope}"


@check("one abnormal sample is 'unsure', and unsure takes no action")
def _():
    system, agent = build()
    system.inject_memory_leak()
    system.advance()
    incident = agent.step()
    assert incident is not None
    assert incident.diagnosis.cause != "memory_spike", incident.diagnosis.summary
    assert incident.diagnosis.confidence == "unsure", incident.diagnosis.summary
    assert not incident.diagnosis.actionable
    assert incident.actions == [], "acted on a single sample"
    assert incident.outcome == "observing"
    assert system.restarted == []


@check("a large but FLAT process is never restarted")
def _():
    # The false positive that gets an agent switched off: a cache warming up
    # looks exactly like the first sample of a leak.
    system, agent = build()
    system.inject_memory_spike(process="postgres", extra_mb=1400.0)
    opened, acted = run_until(agent, system, ticks=30)
    assert acted is None, f"restarted a stable process: {acted and acted.report()}"
    assert system.restarted == []
    causes = {i.diagnosis.cause for i in opened}
    assert "memory_spike" in causes, causes


@check("a healthy machine produces no action at all")
def _():
    # Ten seeds, because this failed on 3 of 10 before min_scale existed:
    # sshd drifting 0.1% -> 1.0% CPU scored 3.1 deviations and proposed a
    # restart. A z-score with a near-zero denominator is not a detector.
    for seed in range(1, 11):
        system, agent = build(seed=seed)
        opened, acted = run_until(agent, system, ticks=30)
        assert acted is None, \
            f"seed {seed}: acted on a healthy machine -- {acted.trigger}"


@check("min_scale floors the deviation in the metric's own units")
def _():
    assert min_scale("proc:sshd:memory") == 15.0
    assert min_scale("proc:sshd:cpu") == 1.0
    assert min_scale("cpu_percent") == 1.0
    base = monitor_mod.Baseline()
    for _ in range(WARMUP_SAMPLES + 5):
        base.observe("proc:sshd:cpu", 0.1)
    # Without the floor this is a division by ~1e-6 and the z-score explodes.
    assert abs(base.z_score("proc:sshd:cpu", 1.0)) < 1.0, base.z_score("proc:sshd:cpu", 1.0)
    assert base.z_score("proc:sshd:cpu", 5.0) > 3.0


# --------------------------------------------------------------------------
# detection: baselines, warmup, backstop
# --------------------------------------------------------------------------

@check("no anomaly is reported before the baseline has warmed up")
def _():
    system = SimulatedSystem()
    monitor = Monitor(system)
    for _ in range(WARMUP_SAMPLES - 2):
        monitor.collect()
        system.advance()
    system.inject_cpu_spike()
    system.advance()
    assert monitor.anomalies(monitor.collect()) == [], \
        "fired before it had any idea what normal looks like"


@check("a leak is caught well below any 90% threshold")
def _():
    system, agent = build()
    system.inject_memory_leak()
    opened, acted = run_until(agent, system)
    at = acted.before["memory"]
    assert at < 60.0, f"only noticed at {at:.1f}% memory"


@check("the absolute ceiling fires even when the baseline learned the fault")
def _():
    # A baseline trained through a slow fill calls the fill normal. The
    # backstop is the answer to a detector that has been taught to ignore the
    # thing it exists to catch.
    system = SimulatedSystem()
    monitor = Monitor(system)
    system.disk_used_mb = system.disk_total * 0.96
    for _ in range(WARMUP_SAMPLES + 20):
        monitor.collect()
        system.advance()
    found = [a for a in monitor.anomalies() if a.metric == "disk_percent"]
    assert found and found[0].severity == "critical", found


# --------------------------------------------------------------------------
# safety: the invariant
# --------------------------------------------------------------------------

@check("INVARIANT: nothing reaches the system except through Policy.check")
def _():
    # dry_run is the cheapest proof of the routing: if any execution path
    # bypassed the gate, something here would still have happened.
    system, agent = build(dry_run=True)
    system.inject_memory_leak()
    run_until(agent, system, ticks=30, stop=lambda inc: False)
    assert system.restarted == [], "dry-run executed a restart"
    assert system.killed == set()
    assert any(i.actions for i in agent.incidents), "nothing was even proposed"
    assert all(not a.executed for i in agent.incidents for a in i.actions)


@check("a protected process is never killable or restartable")
def _():
    policy = Policy(auto_remediate=True, dry_run=False,
                    allowed_services={"sshd", "systemd"})
    for name in ("sshd", "systemd", "lsass.exe", "SSHD"):
        assert not policy.check("kill_process", name).allowed, name
    assert not policy.check("kill_process", "1").allowed
    assert not policy.check("kill_process", str(os.getpid())).allowed


@check("destructive actions need approval even with --auto-remediate")
def _():
    policy = Policy(auto_remediate=True, dry_run=False)
    for action, capability in CAPABILITIES.items():
        if capability.tier != DESTRUCTIVE:
            continue
        decision = policy.check(action, "/var/tmp/x" if "file" in action else "1500")
        assert not decision.allowed and decision.needs_approval, action
        assert policy.check(action, "/var/tmp/x" if "file" in action else "1500",
                            approved=True).allowed, action


@check("an unlisted service cannot be restarted")
def _():
    policy = Policy(auto_remediate=True, dry_run=False, allowed_services={"nginx"})
    assert policy.check("restart_service", "nginx").allowed
    assert not policy.check("restart_service", "postgres").allowed


@check("an invented action is refused, and the refusal names the real ones")
def _():
    decision = Policy().check("reboot_machine", "now")
    assert not decision.allowed
    assert "kill_process" in decision.reason and "restart_service" in decision.reason


@check("protected paths and shallow wildcards are refused")
def _():
    policy = Policy()
    for target in ("/", "/etc", "/var/lib", "C:\\Windows", "/usr/bin", "/*"):
        assert not policy.check("delete_files", target, approved=True).allowed, target
    assert policy.check("delete_files", "/var/tmp/build-cache",
                        approved=True).allowed


@check("the capability table is a finite list that fits on a screen")
def _():
    rows = describe_capabilities()
    assert len(rows) == len(CAPABILITIES) <= 20
    assert {r["tier"] for r in rows} == {"safe", "disruptive", "destructive"}


# --------------------------------------------------------------------------
# the loop: verify, and the incident record
# --------------------------------------------------------------------------

@check("verify re-measures instead of trusting the action")
def _():
    system, agent = build()
    system.inject_disk_pressure()
    opened, acted = run_until(agent, system)
    assert acted.verified is True
    assert acted.after["disk"] < acted.before["disk"]


@check("an incident with no executed action is not reported as verified")
def _():
    system, agent = build(auto_remediate=False)     # restart is gated
    system.inject_memory_leak()
    opened, _ = run_until(agent, system, ticks=30,
                          stop=lambda inc: inc.outcome == "awaiting-approval")
    blocked = [i for i in opened if i.outcome == "awaiting-approval"]
    assert blocked, [i.outcome for i in opened]
    assert blocked[0].verified is None
    assert system.restarted == []


@check("a watched condition stays one incident, not one per pass")
def _():
    system, agent = build()
    system.inject_memory_spike()
    run_until(agent, system, ticks=30, stop=lambda inc: False)
    observing = [i for i in agent.incidents if i.outcome == "observing"]
    assert len(observing) <= 2, [i.diagnosis.cause for i in observing]
    assert sum(i.repeats for i in observing) > 0, "repeats were never counted"


@check("the incident report answers what/what-was-done/did-it-work")
def _():
    system, agent = build()
    system.inject_cpu_spike()
    opened, acted = run_until(agent, system)
    text = acted.report()
    assert text.isascii(), "non-ascii in a report that has to print on Windows"
    for expected in ("trigger", "cause", "outcome", "verified: yes", "restart_service"):
        assert expected in text, expected


@check("summary counts observing apart from unresolved")
def _():
    system, agent = build()
    system.inject_memory_spike()
    run_until(agent, system, ticks=30, stop=lambda inc: False)
    summary = agent.summary()
    assert summary["observing"] >= 1
    assert summary["unresolved"] == 0, "declining to act was scored as a failure"
    assert summary["actions_executed"] == 0


# --------------------------------------------------------------------------
# Phase 5: prove the suite fails on the code it was written against
# --------------------------------------------------------------------------

def _old_diagnose_rules(self, sample, anomalies):
    """The rule as it was when the bug was live: system memory, fixed window."""
    by_metric = {a.metric: a for a in anomalies}
    if by_metric.get("disk_percent") or [a for a in anomalies if a.metric.endswith(":cpu")]:
        return _new_rules(self, sample, anomalies)

    process_memory = [a for a in anomalies if a.metric.endswith(":memory")]
    if not process_memory:
        return _new_rules(self, sample, anomalies)

    worst = process_memory[0]
    name = worst.evidence.get("name", "?")
    slope = self.monitor.trend("memory_percent", agent_mod.TREND_SAMPLES)
    if slope is not None and slope > 0.25:
        return Diagnosis("memory_leak", "likely", "climbing",
                         {"process": name, "pid": worst.evidence.get("pid")})
    return Diagnosis("memory_spike", "likely", "not climbing",
                     {"process": name, "pid": worst.evidence.get("pid")})


_new_rules = HealingAgent._diagnose_rules


def verify_suite_is_not_decorative():
    """Run the leak tests against the old rule. They must fail."""
    leak_tests = [(n, f) for n, f in _results
                  if "leak" in n or "unsure" in n or "FLAT" in n]
    HealingAgent._diagnose_rules = _old_diagnose_rules
    try:
        failures = 0
        for name, fn in leak_tests:
            try:
                fn()
            except AssertionError:
                failures += 1
    finally:
        HealingAgent._diagnose_rules = _new_rules
    return failures, len(leak_tests)


def main():
    failed = 0
    for name, fn in _results:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}\n      {exc}")
    print(f"\n{len(_results) - failed}/{len(_results)} passed")

    caught, total = verify_suite_is_not_decorative()
    print(f"against the pre-fix diagnosis rule: {caught}/{total} of the leak "
          f"tests fail, as they must")
    if caught == 0:
        print("the suite does not detect the bug it was written for")
        failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
