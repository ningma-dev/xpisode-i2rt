"""Whether this host is set up: ``python -m xpisode_i2rt.doctor``. Changes nothing."""

from __future__ import annotations

import os
import resource
import subprocess
import sys
from pathlib import Path

from xpisode_i2rt.core import default_binary

FIFO_PRIORITY = 80


def _gets_fifo() -> bool:
    """Whether a process started from here really gets it: a limit is only what should happen."""
    trial = f"import os; os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param({FIFO_PRIORITY}))"
    return subprocess.run([sys.executable, "-c", trial], capture_output=True, check=False).returncode == 0


def can_be_realtime(binary: Path | None = None) -> bool:
    """Whether the controller will get SCHED_FIFO: as any process here, or by the binary's own right."""
    if _gets_fifo():
        return True
    try:
        os.getxattr(binary or default_binary(), "security.capability")
    except OSError:
        return False
    return True


def main() -> int:
    ready = True

    def line(ok: bool, what: str, fix: str = "") -> None:
        nonlocal ready
        ready = ready and ok
        print(f"{'ok  ' if ok else 'FAIL'} {what}" + (f"\n       {fix}" if fix and not ok else ""))

    binary = default_binary()
    line(binary.exists(), f"controller built: {binary}", "cargo build --release")
    limit = resource.getrlimit(resource.RLIMIT_RTPRIO)[0]
    line(
        can_be_realtime(),
        f"real-time priority from this terminal (its limit is {limit})",
        "sudo scripts/setup_host.sh, then reboot",
    )
    memory = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    line(
        memory == resource.RLIM_INFINITY or memory >= 256 << 20,
        "memory locking allowed",
        "sudo scripts/setup_host.sh, then reboot",
    )
    networks = sorted(Path("/sys/class/net").glob("can*"))
    for net in networks:
        line((net / "operstate").read_text().strip() != "down", f"{net.name} is up", "ip link set up")
    if not networks:
        line(False, "CAN interfaces", "none found")
    governors = {p.read_text().strip() for p in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor")}
    print(f"info CPU governor: {', '.join(sorted(governors)) or 'unknown'} (performance is steadier)")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
