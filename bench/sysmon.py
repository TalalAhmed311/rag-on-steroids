"""Sample process + host resource counters during bench runs."""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil


def _read_diskstats(device: str) -> Optional[tuple[int, int]]:
    """Return (sectors_read, reads_completed) for device basename, or None."""
    path = "/proc/diskstats"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 14:
                continue
            if parts[2] == device:
                # reads_completed, sectors_read
                return int(parts[3]), int(parts[5])
    return None


@dataclass
class SysSample:
    ts: float
    rss_gb: float
    cpu_pct: float
    disk_reads: Optional[int] = None
    disk_sectors_read: Optional[int] = None


@dataclass
class SysMonitor:
    interval_s: float = 0.5
    disk_device: Optional[str] = None  # e.g. "xvdf"
    samples: list[SysSample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None
    _proc: psutil.Process = field(default_factory=psutil.Process)

    def start(self) -> None:
        self.samples.clear()
        self._stop.clear()
        self._proc.cpu_percent(None)  # prime
        self._thread = threading.Thread(target=self._loop, name="sysmon", daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.summary()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            rss_gb = self._proc.memory_info().rss / (1024**3)
            cpu = self._proc.cpu_percent(None)
            reads = sectors = None
            if self.disk_device:
                stats = _read_diskstats(self.disk_device)
                if stats:
                    reads, sectors = stats
            self.samples.append(
                SysSample(
                    ts=time.time(),
                    rss_gb=rss_gb,
                    cpu_pct=cpu,
                    disk_reads=reads,
                    disk_sectors_read=sectors,
                )
            )

    def summary(self) -> dict:
        if not self.samples:
            return {
                "peak_rss_gb": 0.0,
                "avg_cpu_pct": 0.0,
                "disk_reads_delta": None,
                "disk_read_mb_delta": None,
                "n_samples": 0,
            }
        peak = max(s.rss_gb for s in self.samples)
        avg_cpu = sum(s.cpu_pct for s in self.samples) / len(self.samples)
        disk_reads_delta = disk_mb = None
        if (
            self.disk_device
            and self.samples[0].disk_reads is not None
            and self.samples[-1].disk_reads is not None
        ):
            disk_reads_delta = self.samples[-1].disk_reads - self.samples[0].disk_reads
            sectors = (self.samples[-1].disk_sectors_read or 0) - (
                self.samples[0].disk_sectors_read or 0
            )
            disk_mb = sectors * 512 / (1024**2)
        return {
            "peak_rss_gb": round(peak, 4),
            "avg_cpu_pct": round(avg_cpu, 2),
            "disk_reads_delta": disk_reads_delta,
            "disk_read_mb_delta": None if disk_mb is None else round(disk_mb, 4),
            "n_samples": len(self.samples),
        }
