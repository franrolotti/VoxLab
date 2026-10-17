"""Basque (euskara) voices from HiTZ/Aholab: VITS models run by the aHoTTS engine.

aHoTTS (https://github.com/hitz-zentroa/aHoTTS) is published only as a Linux
x86-64 binary that bundles the Basque text front end and the ONNX VITS
runtime. VoxLab runs it in a throwaway ``linux/amd64`` Debian container
(emulated with Rosetta/QEMU on other hosts) with the cached files mounted
read-only and networking disabled. The only requirement is Docker.

Files live in ``<model_dir>/ahotts``: the engine and dictionary (~20 MB) and
one ``vits.onnx`` per voice (~113 MB each), fetched on first use. The
Debian image (~120 MB) is kept by Docker (``docker rmi`` to remove it).
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from voxlab.errors import BackendError
from voxlab.tts.base import (
    BackendCapabilities,
    ModelInfo,
    SpeakerInfo,
    SynthesisRequest,
    SynthesisResult,
    TTSBackend,
)

log = logging.getLogger(__name__)

# Pinned for reproducibility and integrity; bump deliberately.
ENGINE_REPO = "hitz-zentroa/aHoTTS"
ENGINE_REVISION = "db56c5dd27a692b76cd23f7d784c9c465f6a9a18"
ENGINE_FILES: dict[str, str] = {
    # path in the repo (and in the cache) -> sha256
    "ahotts/tts": "5a1760ba48f0dbc80fbad0b551defad9625b2d220c9de12c821e804556bf112e",
    "libonnxruntime.so.1.13.1": "9ed58001185fe72c9382756100a53f3bf09c716f52959145bb10ee7db35570b1",
    "ahotts/dicts/eu/eu_dicc.dic": (
        "7c152ccb24e932296d3769957c85c446a47f8664bb555bf61e31c332586c2b93"
    ),
    "ahotts/dicts/eu/eu_dic_20220725.dit": (
        "882af095ced2c506c7079b63a54d507c435578426c27b1666804920f28c59223"
    ),
}

# speaker id -> (Hugging Face repo, revision, sha256 of vits.onnx, gender)
VOICES: dict[str, tuple[str, str, str, str]] = {
    "antton": (
        "HiTZ/TTS-eu_antton",
        "55a8b9259027f3344602936d3ae5373c7372d0bd",
        "c049d4bf573f05abd005c49d3a319d1a4387cb3acaebe367a910682e55a3424b",
        "male",
    ),
    "maider": (
        "HiTZ/TTS-eu_maider",
        "dea86eeb9038e70104ad6b0b7b8e9ae9fbc3247e",
        "db8fd123eb1040eabf18850e3d6be372c8a0dc36eb3c1ad2c0f40871a0f8bbb7",
        "female",
    ),
}
DEFAULT_SPEAKER = "maider"
VOICE_MB = 113
ENGINE_MB = 20

IMAGE_DIGEST = "sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587"
IMAGE = f"debian:bookworm-slim@{IMAGE_DIGEST}"
PLATFORM = "linux/amd64"

# The engine reads ISO-8859-1. Map common typographic characters to it first.
_LATIN1_FIXES = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "…": "...",
    }
)


class AhoTTSBackend(TTSBackend):
    name = "ahotts"
    capabilities = BackendCapabilities(
        voice_cloning=False,
        languages=("eu",),
        native_params=frozenset(),
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ready = False

    @classmethod
    def is_installed(cls) -> bool:
        return shutil.which("docker") is not None

    # --- files --------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self.model_dir / "ahotts"

    @property
    def engine_dir(self) -> Path:
        return self.root / "engine"

    def voice_file(self, speaker: str) -> Path:
        return self.root / "voices" / speaker / "vits.onnx"

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            backend=self.name,
            model_id="HiTZ/TTS-eu_{antton,maider}",
            variant="vits",
            license="Apache-2.0 / CC BY 4.0",
            download_mb=ENGINE_MB + VOICE_MB * len(VOICES),
            url="https://huggingface.co/HiTZ/TTS-eu_antton",
        )

    def _engine_ready(self) -> bool:
        return all((self.engine_dir / path).is_file() for path in ENGINE_FILES)

    def is_downloaded(self) -> bool:
        return self._engine_ready() and all(self.voice_file(v).is_file() for v in VOICES)

    def download(self) -> None:
        """Engine, every voice and the container image."""
        self._download_engine()
        for speaker in VOICES:
            self._download_voice(speaker)
        self._pull_image()

    def _download_engine(self) -> None:
        log.info("Downloading the aHoTTS engine (~%d MB) to %s", ENGINE_MB, self.engine_dir)
        base = f"https://github.com/{ENGINE_REPO}/raw/{ENGINE_REVISION}"
        for path, sha256 in ENGINE_FILES.items():
            target = self.engine_dir / path
            if not target.is_file():
                fetch(f"{base}/{path}", target, sha256)
        (self.engine_dir / "ahotts" / "tts").chmod(0o755)

    def _download_voice(self, speaker: str) -> None:
        target = self.voice_file(speaker)
        if target.is_file():
            return
        repo, revision, sha256, _ = VOICES[speaker]
        log.info("Downloading Basque voice %r (~%d MB) to %s", speaker, VOICE_MB, target.parent)
        fetch(f"https://huggingface.co/{repo}/resolve/{revision}/vits.onnx", target, sha256)

    def _pull_image(self) -> None:
        if self._docker("image", "inspect", IMAGE).returncode == 0:
            return
        log.info("Pulling the %s container image (~120 MB)", PLATFORM)
        result = self._docker("pull", "--platform", PLATFORM, IMAGE)
        if result.returncode != 0:
            raise BackendError(f"docker pull failed: {_tail(result.stderr)}")

    # --- inference ----------------------------------------------------------

    def load(self) -> None:
        if self._ready:
            return
        if not self.is_installed():
            raise BackendError(
                "Basque voices need Docker: install Docker Desktop (or colima) and start it"
            )
        if self._docker("info").returncode != 0:
            raise BackendError("Docker is installed but not running: start Docker Desktop")
        if not self._engine_ready():
            self._require_download()
            self._download_engine()
        if self._docker("image", "inspect", IMAGE).returncode != 0:
            self._require_download()
            self._pull_image()
        self._ready = True

    def _require_download(self) -> None:
        if not self.allow_download:
            raise BackendError(
                "aHoTTS files not found and auto-download is disabled. "
                "Run `voxlab models --download --backend ahotts` first."
            )

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        if request.reference_audio is not None:
            raise BackendError("aHoTTS does not support voice cloning (reference audio)")
        if not self.supports_language(request.language):
            raise BackendError(f"aHoTTS voices only speak Basque (eu), not {request.language!r}")
        speaker = request.speaker or DEFAULT_SPEAKER
        if speaker not in VOICES:
            raise BackendError(
                f"Unknown aHoTTS speaker {speaker!r}. Available: {', '.join(VOICES)}"
            )
        text = encode_text(request.text)
        self.load()
        if not self.voice_file(speaker).is_file():
            self._require_download()
            self._download_voice(speaker)

        with tempfile.TemporaryDirectory(prefix="voxlab-ahotts-") as out:
            command = [
                *("docker", "run", "--rm", "-i", "--platform", PLATFORM, "--network", "none"),
                *("-v", f"{self.engine_dir}:/opt/ahotts:ro"),
                *("-v", f"{self.voice_file(speaker).parent}:/voice:ro"),
                *("-v", f"{out}:/out", "-w", "/opt/ahotts"),
                *("-e", "LD_LIBRARY_PATH=/opt/ahotts", IMAGE),
                *("./ahotts/tts", "-Lang=eu", "-Method=Vits", "-HDic=./ahotts/dicts/eu/eu_dicc"),
                *("-voice_path=/voice", "/out/line.wav"),
            ]
            result = subprocess.run(command, input=text, capture_output=True, check=False)
            wav = Path(out) / "line.wav"
            if result.returncode != 0 or not wav.is_file():
                raise BackendError(
                    f"aHoTTS failed on {request.text[:40]!r}: "
                    f"{_tail(result.stderr) or _tail(result.stdout) or 'no audio produced'}"
                )
            import soundfile as sf

            audio, sample_rate = sf.read(wav, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return SynthesisResult(np.ascontiguousarray(audio), int(sample_rate))

    def speakers(self) -> list[SpeakerInfo]:
        return [SpeakerInfo(id=v, language="eu", gender=VOICES[v][3]) for v in VOICES]

    def default_speaker(self, language: str) -> str | None:
        return DEFAULT_SPEAKER

    def runtime_info(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "engine": f"{ENGINE_REPO}@{ENGINE_REVISION[:12]}",
            "image": IMAGE,
            "platform": PLATFORM,
        }

    @staticmethod
    def _docker(*args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(["docker", *args], capture_output=True, check=False)


def encode_text(text: str) -> bytes:
    """Text as the ISO-8859-1 bytes the engine reads, or a clear error."""
    text = text.translate(_LATIN1_FIXES)
    bad = sorted({ch for ch in text if ord(ch) > 0xFF})
    if bad:
        raise BackendError(
            f"aHoTTS cannot read these characters: {' '.join(bad)} (in {text[:40]!r})"
        )
    return text.encode("latin-1")


def fetch(url: str, target: Path, sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as fh:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                fh.write(chunk)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise BackendError(f"Download failed ({url}): {exc}") from exc
    if digest.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise BackendError(f"Checksum mismatch for {url}; refusing to use it")
    tmp.replace(target)


def _tail(output: bytes, lines: int = 3) -> str:
    text = output.decode("utf-8", errors="replace").strip()
    return " | ".join(text.splitlines()[-lines:])
