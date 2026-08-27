"""Metric collection and anomaly detection, with a simulated system for tests.

WHY A SIMULATED SYSTEM
----------------------
The agent's job is to notice a machine going wrong and fix it. Testing that
against a real machine means either waiting for a real memory leak, or
deliberately causing one on the machine running the tests -- so in practice
nobody tests it, and the detection thresholds are guesses that were never
checked against a case they were supposed to catch.

`SimulatedSystem` produces a leak, a runaway process, disk pressure or a crash
loop on demand, deterministically. The agent cannot tell it apart from
`RealSystem`, which is a thin psutil wrapper.

DETECTION: BASELINES, NOT THRESHOLDS
------------------------------------
"Alert if memory > 90%" is wrong in both directions. A build server sitting at
92% is fine; a database that normally runs at 40% and is now at 70% has a
problem worth catching two hours before it hits 90.

So each metric keeps a rolling baseline and anomalies are z-scores against it,
with an absolute ceiling only as a backstop for the genuinely urgent. The cost
is a warmup period during which the agent knows nothing and says so, rather
than firing on the first sample it sees.
"""

import os
import random
import time
from collections import deque
from dataclasses import dataclass, field

WARMUP_SAMPLES = 10        # before this, no anomaly is reported
Z_THRESHOLD = 3.0          # standard deviations from the rolling baseline
DISK_CRITICAL = 95.0       # absolute backstop, percent used
MEMORY_CRITICAL = 95.0
MIN_TREND_RUN = 4          # abnormal samples needed before a slope means anything

# Suffix used in the baseline key -> attribute on ProcessInfo.
PROCESS_METRICS = {"memory": "memory_mb", "cpu": "cpu_percent"}


@dataclass
class ProcessInfo:
    pid: int
    name: str
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    status: str = "running"
    created: float = 0.0


@dataclass
class Sample:
    at: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_percent: float
    disk_free_mb: float
    processes: list = field(default_factory=list)
    log_errors: int = 0

    def top_by_memory(self, n=5):
        return sorted(self.processes, key=lambda p: -p.memory_mb)[:n]

    def top_by_cpu(self, n=5):
        return sorted(self.processes, key=lambda p: -p.cpu_percent)[:n]


@dataclass
class Anomaly:
    metric: str
    severity: str            # warning | critical
    value: float
    baseline: float
    z_score: float
    detail: str
    evidence: dict = field(default_factory=dict)


class SimulatedSystem:
    """A machine that can be made to go wrong on request."""

    def __init__(self, clock=None, seed=42, memory_total_mb=16384,
                 disk_total_mb=512000):
        self._t = 0.0
        self.clock = clock or (lambda: self._t)
        self.rng = random.Random(seed)
        self.memory_total = memory_total_mb
        self.disk_total = disk_total_mb

        self.base_cpu = 15.0
        self.base_memory_mb = 4000.0
        self.disk_used_mb = disk_total_mb * 0.55

        self.faults = set()
        self.leak_mb_per_tick = 0.0
        self.leaked_mb = 0.0
        self.disk_fill_mb_per_tick = 0.0
        self.log_error_rate = 0
        self.spike_mem_mb = 0.0
        self.killed = set()
        self.restarted = []

        self.processes = [
            ProcessInfo(1, "systemd", 0.1, 12),
            ProcessInfo(410, "sshd", 0.1, 18),
            ProcessInfo(900, "postgres", 6.0, 1400),
            ProcessInfo(1201, "nginx", 2.0, 90),
            ProcessInfo(1500, "app-worker", 8.0, 700),
            ProcessInfo(1777, "metrics-agent", 1.0, 60),
        ]

    def advance(self, ticks=1):
        for _ in range(ticks):
            self._t += 5.0
            if "leak" in self.faults:
                self.leaked_mb += self.leak_mb_per_tick
            if "disk" in self.faults:
                self.disk_used_mb = min(self.disk_total,
                                        self.disk_used_mb + self.disk_fill_mb_per_tick)

    # -- fault injection -------------------------------------------------

    def inject_memory_leak(self, process="app-worker", mb_per_tick=180.0):
        self.faults.add("leak")
        self.leak_mb_per_tick = mb_per_tick
        self.leak_process = process

    def inject_memory_spike(self, process="postgres", extra_mb=1400.0):
        """A process that jumps to a large-but-STABLE footprint and stays there.

        The counterpart to `inject_memory_leak`, and the one that catches false
        positives: a cache warming up or a batch job loading a dataset looks
        exactly like the first sample of a leak. Restarting it is a
        self-inflicted outage, so the agent has to be measured on this case
        too, not only on the one it is supposed to fix.
        """
        self.faults.add("mem_spike")
        self.spike_mem_process = process
        self.spike_mem_mb = extra_mb

    def inject_cpu_spike(self, process="app-worker", cpu=340.0):
        self.faults.add("cpu")
        self.spike_process = process
        self.spike_cpu = cpu

    def inject_disk_pressure(self, mb_per_tick=6000.0):
        self.faults.add("disk")
        self.disk_fill_mb_per_tick = mb_per_tick

    def inject_log_errors(self, rate=40):
        self.faults.add("logs")
        self.log_error_rate = rate

    def clear_faults(self):
        self.faults.clear()
        self.spike_mem_mb = 0.0
        self.leaked_mb = 0.0
        self.leak_mb_per_tick = 0.0
        self.disk_fill_mb_per_tick = 0.0
        self.log_error_rate = 0

    # -- the system interface --------------------------------------------

    def sample(self):
        processes = []
        for p in self.processes:
            if p.pid in self.killed:
                continue
            info = ProcessInfo(p.pid, p.name, p.cpu_percent, p.memory_mb,
                               p.status, p.created)
            if "leak" in self.faults and p.name == getattr(self, "leak_process", None):
                info.memory_mb += self.leaked_mb
            if ("mem_spike" in self.faults
                    and p.name == getattr(self, "spike_mem_process", None)):
                info.memory_mb += self.spike_mem_mb
            if "cpu" in self.faults and p.name == getattr(self, "spike_process", None):
                info.cpu_percent = self.spike_cpu
            info.cpu_percent += self.rng.gauss(0, 0.3)
            info.memory_mb += self.rng.gauss(0, 4)
            processes.append(info)

        memory_used = sum(p.memory_mb for p in processes) + self.base_memory_mb
        cpu = min(100.0, sum(p.cpu_percent for p in processes) / 4
                  + self.rng.gauss(0, 1.0))

        return Sample(
            at=self.clock(),
            cpu_percent=max(0.0, cpu),
            memory_percent=100.0 * memory_used / self.memory_total,
            memory_used_mb=memory_used,
            disk_percent=100.0 * self.disk_used_mb / self.disk_total,
            disk_free_mb=self.disk_total - self.disk_used_mb,
            processes=processes,
            log_errors=self.log_error_rate + (self.rng.randint(0, 2)),
        )

    # -- actions the agent can take --------------------------------------

    def kill_process(self, pid):
        pid = int(pid)
        if pid not in {p.pid for p in self.processes}:
            raise LookupError(f"no process {pid}")
        self.killed.add(pid)
        if "leak" in self.faults and any(
                p.pid == pid and p.name == getattr(self, "leak_process", None)
                for p in self.processes):
            self.faults.discard("leak")
            self.leaked_mb = 0.0
        if "cpu" in self.faults and any(
                p.pid == pid and p.name == getattr(self, "spike_process", None)
                for p in self.processes):
            self.faults.discard("cpu")
        return True

    def restart_service(self, name):
        self.restarted.append(name)
        if "leak" in self.faults and name == getattr(self, "leak_process", None):
            self.faults.discard("leak")
            self.leaked_mb = 0.0
        if "cpu" in self.faults and name == getattr(self, "spike_process", None):
            self.faults.discard("cpu")
        return True

    def free_disk(self, mb):
        freed = min(mb, self.disk_used_mb)
        self.disk_used_mb -= freed
        self.faults.discard("disk")
        self.disk_fill_mb_per_tick = 0.0
        return freed


class RealSystem:
    """psutil, behind the same interface. Read-only unless asked otherwise."""

    def __init__(self, clock=time.time):
        try:
            import psutil
        except ImportError:
            raise RuntimeError(
                "psutil is not installed (pip install psutil). Use "
                "SimulatedSystem for tests and development -- it can inject a "
                "memory leak or disk pressure on demand, which a real machine "
                "cannot be asked to do politely."
            )
        self.psutil = psutil
        self.clock = clock

    def sample(self):
        ps = self.psutil
        memory = ps.virtual_memory()
        disk = ps.disk_usage("/" if not _is_windows() else "C:\\")

        processes = []
        for proc in ps.process_iter(["pid", "name", "cpu_percent", "memory_info", "status"]):
            try:
                info = proc.info
                processes.append(ProcessInfo(
                    pid=info["pid"], name=info["name"] or "?",
                    cpu_percent=info["cpu_percent"] or 0.0,
                    memory_mb=(info["memory_info"].rss / 1_048_576)
                    if info["memory_info"] else 0.0,
                    status=info["status"] or "?"))
            except (ps.NoSuchProcess, ps.AccessDenied):
                continue        # a process that vanished mid-scan is normal

        return Sample(
            at=self.clock(), cpu_percent=ps.cpu_percent(interval=None),
            memory_percent=memory.percent, memory_used_mb=memory.used / 1_048_576,
            disk_percent=disk.percent, disk_free_mb=disk.free / 1_048_576,
            processes=processes)

    def kill_process(self, pid):
        self.psutil.Process(int(pid)).terminate()
        return True

    def restart_service(self, name):
        raise NotImplementedError(
            "restarting a real service needs an init-system adapter (systemctl, "
            "sc.exe, launchctl). Not implemented rather than guessed at.")

    def free_disk(self, mb):
        raise NotImplementedError(
            "deleting real files is gated by safety.Policy and needs an explicit "
            "target; there is no 'free N megabytes from somewhere' operation.")


def _is_windows():
    return os.name == "nt"


# --------------------------------------------------------------------------
# anomaly detection
# --------------------------------------------------------------------------

def min_scale(metric):
    """Floor on a metric's deviation, so a z-score cannot divide by ~nothing.

    THE BUG THIS EXISTS FOR
    -----------------------
    A z-score is a ratio, and a ratio with a near-zero denominator is not a
    detector. `sshd` sits at 0.1% CPU and never moves, so its MAD collapses to
    roughly zero -- and the ordinary gaussian jitter that takes it to 1.0% CPU
    scored **3.1 deviations** and opened an incident, on a machine with nothing
    wrong with it. Three healthy runs out of ten produced a remediation
    proposal against `sshd` and `systemd`.

    The fix is not a bigger `Z_THRESHOLD`; that trades these false positives
    for missed real ones on noisier metrics. The fix is to say what change is
    too small to be worth calling an anomaly *in the metric's own units*, and
    floor the denominator there. One percentage point of CPU is noise. Fifteen
    megabytes of RSS is noise. Below that, no multiple of it is a finding.
    """
    return 15.0 if metric.endswith(":memory") else 1.0


class Baseline:
    """A rolling mean and standard deviation per metric.

    Uses a median-based deviation rather than the plain standard deviation:
    once a metric is anomalous it starts inflating its own baseline's variance,
    which raises the bar and hides the very anomaly being tracked. That is the
    detector suppressing itself, and it happens exactly when it matters.
    """

    def __init__(self, window=60):
        self.window = window
        self.values = {}

    def observe(self, metric, value):
        self.values.setdefault(metric, deque(maxlen=self.window)).append(value)

    def stats(self, metric):
        series = self.values.get(metric)
        if not series or len(series) < WARMUP_SAMPLES:
            return None
        ordered = sorted(series)
        n = len(ordered)
        median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
        deviations = sorted(abs(v - median) for v in series)
        mad = deviations[n // 2] if n % 2 else (deviations[n // 2 - 1] + deviations[n // 2]) / 2
        # 1.4826 makes MAD comparable to a standard deviation for normal data.
        return median, max(mad * 1.4826, min_scale(metric))

    def z_score(self, metric, value):
        stats = self.stats(metric)
        if stats is None:
            return None
        median, scale = stats
        return (value - median) / scale

    def ready(self, metric):
        return len(self.values.get(metric, ())) >= WARMUP_SAMPLES


class Monitor:
    """Samples the system and reports anomalies against a learned baseline."""

    def __init__(self, system, window=60):
        self.system = system
        self.baseline = Baseline(window)
        self.history = deque(maxlen=window)

    def collect(self):
        sample = self.system.sample()
        self.history.append(sample)
        for metric in ("cpu_percent", "memory_percent", "disk_percent"):
            self.baseline.observe(metric, getattr(sample, metric))
        for process in sample.processes:
            self.baseline.observe(f"proc:{process.name}:memory", process.memory_mb)
            self.baseline.observe(f"proc:{process.name}:cpu", process.cpu_percent)
        return sample

    def anomalies(self, sample=None):
        sample = sample or (self.history[-1] if self.history else None)
        if sample is None:
            return []

        found = []
        for metric, label in (("cpu_percent", "CPU"), ("memory_percent", "memory"),
                              ("disk_percent", "disk")):
            value = getattr(sample, metric)
            z = self.baseline.z_score(metric, value)

            # Absolute backstop. A machine at 96% memory is in trouble whether
            # or not that is where it usually sits -- a baseline learned during
            # a slow leak would otherwise call the leak normal.
            ceiling = {"memory_percent": MEMORY_CRITICAL,
                       "disk_percent": DISK_CRITICAL}.get(metric)
            if ceiling and value >= ceiling:
                found.append(Anomaly(
                    metric, "critical", round(value, 2),
                    round(self.baseline.stats(metric)[0], 2)
                    if self.baseline.ready(metric) else value,
                    z if z is not None else float("inf"),
                    f"{label} at {value:.1f}%, past the {ceiling:.0f}% ceiling"))
                continue

            if z is not None and z > Z_THRESHOLD:
                median, _ = self.baseline.stats(metric)
                found.append(Anomaly(
                    metric, "warning" if z < Z_THRESHOLD * 2 else "critical",
                    round(value, 2), round(median, 2), round(z, 2),
                    f"{label} at {value:.1f}%, {z:.1f} deviations above its "
                    f"usual {median:.1f}%"))

        found.extend(self._process_anomalies(sample))
        found.sort(key=lambda a: (a.severity != "critical", -abs(a.z_score)))
        return found

    def _process_anomalies(self, sample):
        out = []
        for process in sample.processes:
            for suffix, label, unit in (("memory", "memory", "MB"),
                                        ("cpu", "CPU", "%")):
                key = f"proc:{process.name}:{suffix}"
                value = getattr(process, PROCESS_METRICS[suffix])
                z = self.baseline.z_score(key, value)
                if z is None or z <= Z_THRESHOLD:
                    continue
                median, _ = self.baseline.stats(key)
                out.append(Anomaly(
                    key, "warning" if z < Z_THRESHOLD * 2 else "critical",
                    round(value, 1), round(median, 1), round(z, 2),
                    f"{process.name} (pid {process.pid}) using {value:.0f}{unit} of "
                    f"{label}, {z:.1f} deviations above its usual {median:.0f}{unit}",
                    evidence={"pid": process.pid, "name": process.name}))
        return out

    def trend(self, metric, samples=10):
        """Is this system-wide metric rising, and how fast, in units per sample.

        Note what this cannot do: it is a fixed window over *system* totals, so
        one tick into a fault the window is mostly healthy machine. It is the
        right tool for disk, which fills monotonically over minutes. It is the
        wrong tool for telling a memory leak from a memory spike -- use
        `process_trend` for that, and read the docstring there for why.
        """
        if len(self.history) < samples:
            return None
        return _slope([getattr(s, metric) for s in list(self.history)[-samples:]])

    def process_trend(self, name, metric="memory", samples=8):
        """Slope of ONE process's own series over the run it has been abnormal.

        Returns ``(slope, n_abnormal_samples)``. ``slope`` is None when the
        abnormal run is too short to fit a line to, and the caller must then
        say "unsure" rather than pick a cause.

        THE BUG THIS EXISTS FOR
        -----------------------
        A leak used to be diagnosed as a spike. The anomaly fires on the
        *first* abnormal sample, and the old rule then asked
        ``trend("memory_percent", 8)`` -- an 8-sample window over system memory
        of which 7 samples predate the leak entirely. The slope it returned
        described a step, not a trend, so a climbing process was called "high
        but not climbing" and the remediation was skipped.

        Two things fix it, and both are needed:

        1. **The process's own series**, not system memory. System memory is
           the sum over every process plus a 4GB base; one process climbing is
           a small fraction of it, and any other process moving is noise on the
           signal being measured.
        2. **Only the abnormal run.** Fitting across the healthy samples that
           precede the fault measures the onset step. Fitting across the
           abnormal ones measures whether it is still going, which is the
           actual question.

        Fewer than `MIN_TREND_RUN` abnormal samples is not a failure to report
        -- it is the honest answer. A leak and a spike are *identical* on their
        first sample, and restarting a service on one sample is the false
        positive that gets the agent switched off.
        """
        attribute = PROCESS_METRICS[metric]
        stats = self.baseline.stats(f"proc:{name}:{metric}")
        if stats is None:
            return None, 0
        median, scale = stats
        limit = median + Z_THRESHOLD * scale

        run = []
        for sample in reversed(self.history):
            process = next((p for p in sample.processes if p.name == name), None)
            if process is None or getattr(process, attribute) <= limit:
                break
            run.append(getattr(process, attribute))
        run.reverse()

        if len(run) < MIN_TREND_RUN:
            return None, len(run)
        return _slope(run[-samples:]), len(run)


def _slope(values):
    """Least-squares slope in units per sample. Shared so `trend` and
    `process_trend` cannot disagree about what "rising" means."""
    n = len(values)
    mean_x = (n - 1) / 2
    mean_y = sum(values) / n
    numerator = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    denominator = sum((i - mean_x) ** 2 for i in range(n))
    return numerator / denominator if denominator else 0.0
