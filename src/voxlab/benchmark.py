"""``voxlab benchmark``: speed and memory of the installed backend.

Uses only the model that is already configured; it never downloads extra
models.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from voxlab import __version__
from voxlab.tts.base import SynthesisRequest, TTSBackend

BENCHMARK_SENTENCES: dict[str, list[str]] = {
    "es": [
        "¿Me recibes?",
        "Todos los sistemas funcionando.",
        "La tormenta se acercaba desde el oeste, y el faro llevaba tres noches apagado.",
    ],
    "en": [
        "Do you copy?",
        "All systems are operational.",
        "The storm rolled in from the west, and the lighthouse had been dark for three nights.",
    ],
}


@dataclass
class SentenceResult:
    language: str
    text: str
    seconds: float
    audio_seconds: float

    @property
    def rtf(self) -> float:
        return self.seconds / self.audio_seconds if self.audio_seconds else float("inf")


@dataclass
class BenchmarkReport:
    system: dict[str, Any]
    runtime: dict[str, Any]
    load_seconds: float
    results: list[SentenceResult] = field(default_factory=list)
    peak_memory_mb: float | None = None

    @property
    def total_seconds(self) -> float:
        return sum(r.seconds for r in self.results)

    @property
    def total_audio_seconds(self) -> float:
        return sum(r.audio_seconds for r in self.results)

    @property
    def rtf(self) -> float:
        audio = self.total_audio_seconds
        return self.total_seconds / audio if audio else float("inf")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for item, result in zip(data["results"], self.results, strict=True):
            item["rtf"] = round(result.rtf, 3)
        data.update(
            total_seconds=self.total_seconds,
            total_audio_seconds=self.total_audio_seconds,
            rtf=self.rtf,
        )
        return data


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process (None where unsupported, e.g. Windows)."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def system_info() -> dict[str, Any]:
    return {
        "voxlab": __version__,
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _total_ram_gb(),
    }


def _cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or "unknown"


def _total_ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        return None


def run_benchmark(
    backend: TTSBackend, languages: list[str] | None = None, runs: int = 1
) -> BenchmarkReport:
    """Load the backend, warm it up, then time each benchmark sentence."""
    languages = [
        lang for lang in (languages or list(BENCHMARK_SENTENCES)) if backend.supports_language(lang)
    ]
    started = time.perf_counter()
    backend.load()
    load_seconds = time.perf_counter() - started

    # Warm-up so one-off initialisation is not counted as synthesis time.
    backend.synthesize(SynthesisRequest(text="OK.", language=languages[0]))

    report = BenchmarkReport(
        system=system_info(), runtime=backend.runtime_info(), load_seconds=load_seconds
    )
    for language in languages:
        for text in BENCHMARK_SENTENCES.get(language, []):
            for _ in range(runs):
                t0 = time.perf_counter()
                result = backend.synthesize(SynthesisRequest(text=text, language=language))
                report.results.append(
                    SentenceResult(
                        language=language,
                        text=text,
                        seconds=time.perf_counter() - t0,
                        audio_seconds=result.duration,
                    )
                )
    report.peak_memory_mb = peak_memory_mb()
    return report
