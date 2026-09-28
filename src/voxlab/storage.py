"""Disk usage and cleanup of downloaded models."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from voxlab.errors import VoxLabError

log = logging.getLogger(__name__)


def dir_size(path: Path) -> int:
    """Total size in bytes of all files under ``path`` (0 if missing)."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def _check_safe_to_delete(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in forbidden or len(resolved.parts) < 3:
        raise VoxLabError(f"Refusing to delete {resolved}: storage.model_dir looks unsafe")
    if (resolved / ".git").exists() or (resolved / "pyproject.toml").exists():
        raise VoxLabError(f"Refusing to delete {resolved}: it looks like a project folder")
    return resolved


def clean_model_dir(model_dir: Path, dry_run: bool = False) -> int:
    """Delete the VoxLab model cache. Returns the number of bytes freed."""
    target = _check_safe_to_delete(model_dir)
    if not target.exists():
        return 0
    size = dir_size(target)
    if dry_run:
        return size
    shutil.rmtree(target)
    log.info("Removed %s (%s)", target, human_size(size))
    return size
