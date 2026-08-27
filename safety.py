"""What the agent is allowed to do, and the gate everything passes through.

THE INVARIANT
-------------
**No action reaches the system unless it is on the allowlist, and every
destructive action is either reversible or blocked pending human approval.**

This is the first file in the project rather than the last, because the whole
premise -- an autonomous agent that kills processes and deletes files -- is
only defensible if the answer to "what could it possibly do?" is a finite list
that fits on a screen.

The failure being prevented is not a crash. It is an agent that reasons its
way, one plausible step at a time, into `kill -9` on the database because the
database was using the most memory. Every individual step looks sound. The
outcome is an outage caused by the thing that was supposed to prevent one.

So capability is **not** derived from reasoning. It is a static table, checked
before execution, and the reasoning layer can only select from it. A model
that hallucinates an action gets a refusal, not an execution.

THREE TIERS
-----------
    safe          reversible, no user impact. Runs automatically.
                  (log rotation, cache clearing, reading anything)
    disruptive    reversible, but users notice. Runs automatically only when
                  `--auto-remediate` is set; otherwise proposed.
                  (restarting a service)
    destructive   irreversible or unsafe to guess at. ALWAYS requires explicit
                  human approval, even in auto mode. No flag disables this.
                  (killing a process, deleting files, patching a config)

PROTECTED PROCESSES
-------------------
A separate, harder rule: some processes are never touchable at any tier, by
any path. Killing `systemd`, `sshd` or the agent's own PID turns a degraded
machine into an unreachable one -- and losing SSH means losing the ability to
fix what the agent just did.
"""

import os
import re
from dataclasses import dataclass, field

SAFE = "safe"
DISRUPTIVE = "disruptive"
DESTRUCTIVE = "destructive"

TIER_ORDER = {SAFE: 0, DISRUPTIVE: 1, DESTRUCTIVE: 2}

# Never killable, never restartable, at any tier. Matched case-insensitively
# against the process name. Losing any of these makes the machine harder to
# recover than whatever the agent was trying to fix.
PROTECTED_PROCESSES = {
    "systemd", "init", "kernel", "kthreadd", "sshd", "ssh-agent",
    "launchd", "svchost.exe", "csrss.exe", "wininit.exe", "winlogon.exe",
    "services.exe", "lsass.exe", "smss.exe", "system", "registry",
    "dockerd", "containerd", "kubelet",
}

# Paths that may never be deleted from, however much space they would free.
PROTECTED_PATHS = (
    "/", "/bin", "/sbin", "/usr", "/etc", "/boot", "/dev", "/proc", "/sys",
    "/lib", "/lib64", "/var/lib", "/home", "/root",
    "c:\\windows", "c:\\program files", "c:\\users",
)


class ActionRefused(Exception):
    """The gate said no. Always carries the reason, because a refusal nobody
    can explain is indistinguishable from a bug."""


@dataclass(frozen=True)
class Capability:
    name: str
    tier: str
    description: str
    reversible: bool
    # What must be true before this may run at all. Checked by `check()`.
    requires_target: bool = True


CAPABILITIES = {
    # -- safe: reversible, invisible to users
    "read_metrics": Capability(
        "read_metrics", SAFE, "sample CPU, memory, disk and process table",
        reversible=True, requires_target=False),
    "read_logs": Capability(
        "read_logs", SAFE, "read the tail of a log file", reversible=True),
    "rotate_logs": Capability(
        "rotate_logs", SAFE, "compress and roll a log file", reversible=True),
    "clear_cache": Capability(
        "clear_cache", SAFE, "delete regenerable cache files", reversible=True),
    "write_report": Capability(
        "write_report", SAFE, "record an incident report",
        reversible=True, requires_target=False),

    # -- disruptive: reversible, but users notice
    "restart_service": Capability(
        "restart_service", DISRUPTIVE, "stop and start a managed service",
        reversible=True),
    "throttle_process": Capability(
        "throttle_process", DISRUPTIVE, "lower a process's scheduling priority",
        reversible=True),

    # -- destructive: irreversible. Human approval, always.
    "kill_process": Capability(
        "kill_process", DESTRUCTIVE, "terminate a process", reversible=False),
    "delete_files": Capability(
        "delete_files", DESTRUCTIVE, "remove files to free disk space",
        reversible=False),
    "patch_config": Capability(
        "patch_config", DESTRUCTIVE, "change a value in a configuration file",
        reversible=False),
}


@dataclass
class Decision:
    allowed: bool
    reason: str
    tier: str = SAFE
    needs_approval: bool = False
    capability: Capability = None


@dataclass
class Policy:
    """What this agent instance may do. Deliberately restrictive by default."""

    auto_remediate: bool = False        # allow DISRUPTIVE without asking
    dry_run: bool = True                # describe, never execute
    max_actions_per_incident: int = 5
    protected_processes: set = field(default_factory=lambda: set(PROTECTED_PROCESSES))
    protected_paths: tuple = PROTECTED_PATHS
    allowed_services: set = field(default_factory=set)   # empty = none allowed

    def check(self, action, target=None, approved=False):
        """The single gate. Every action goes through here before execution.

        Returns a Decision. Never raises for a refusal -- a refused action is
        a normal outcome that gets recorded, not an exception that unwinds the
        loop.
        """
        capability = CAPABILITIES.get(action)
        if capability is None:
            # A model asked for something that does not exist. This is the
            # hallucination case, and the answer is a refusal naming what IS
            # available, so the next attempt can be correct.
            return Decision(False,
                            f"unknown action {action!r}; allowed: "
                            f"{', '.join(sorted(CAPABILITIES))}")

        if capability.requires_target and not target:
            return Decision(False, f"{action} needs a target", capability.tier,
                            capability=capability)

        if action in ("kill_process", "throttle_process"):
            refusal = self._check_process(target)
            if refusal:
                return Decision(False, refusal, capability.tier, capability=capability)

        if action == "delete_files":
            refusal = self._check_path(target)
            if refusal:
                return Decision(False, refusal, capability.tier, capability=capability)

        if action == "restart_service":
            if target not in self.allowed_services:
                return Decision(
                    False,
                    f"service {target!r} is not in the allowed set "
                    f"({sorted(self.allowed_services) or 'empty'}); "
                    f"restarting an unlisted service is how an agent takes down "
                    f"something it was never meant to manage",
                    capability.tier, capability=capability)

        if capability.tier == DESTRUCTIVE and not approved:
            return Decision(
                False,
                f"{action} on {target!r} is destructive and needs explicit human "
                f"approval; --auto-remediate does not cover it",
                capability.tier, needs_approval=True, capability=capability)

        if capability.tier == DISRUPTIVE and not (self.auto_remediate or approved):
            return Decision(
                False,
                f"{action} on {target!r} is disruptive; run with --auto-remediate "
                f"or approve it explicitly",
                capability.tier, needs_approval=True, capability=capability)

        return Decision(True, "permitted by policy", capability.tier,
                        capability=capability)

    def _check_process(self, target):
        """Protected names and the agent's own PID.

        Both halves matter. The name check stops `kill sshd`; the PID check
        stops the agent killing itself, which sounds absurd until an agent
        looking for the highest-memory Python process finds itself.
        """
        name = str(target)
        if name.isdigit():
            if int(name) == os.getpid():
                return "refusing to kill the agent's own process"
            if int(name) <= 1:
                return f"refusing to kill PID {name}: that is init/systemd"
            return None

        base = re.sub(r"\.(exe|bin)$", "", name.lower())
        if base in {p.lower().replace(".exe", "") for p in self.protected_processes}:
            return (f"{name!r} is a protected process; stopping it makes the machine "
                    f"harder to recover than whatever it was doing wrong")
        return None

    def _check_path(self, target):
        path = str(target).replace("\\", "/").rstrip("/").lower()
        if not path:
            return "refusing to delete from an empty path"
        for protected in self.protected_paths:
            normalised = protected.replace("\\", "/").rstrip("/").lower()
            if path == normalised:
                return f"{target!r} is a protected path"
            if not normalised or normalised == "/":
                continue
            if path.startswith(normalised + "/") and path.count("/") <= normalised.count("/") + 1:
                return (f"{target!r} is directly inside the protected path "
                        f"{protected!r}; deleting here is not something to guess at")
        if "*" in path and path.count("/") < 3:
            return f"refusing a wildcard delete this close to the root: {target!r}"
        return None


def describe_capabilities():
    """The full list, for `--capabilities` and for a model's system prompt.

    A model that can see the whole table has no reason to invent an action,
    and one that invents one anyway gets a refusal naming this list.
    """
    rows = []
    for name in sorted(CAPABILITIES, key=lambda n: (TIER_ORDER[CAPABILITIES[n].tier], n)):
        c = CAPABILITIES[name]
        rows.append({"action": c.name, "tier": c.tier,
                     "reversible": c.reversible, "description": c.description})
    return rows
