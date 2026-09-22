"""Compatibility loader for OrcaLab CameraSensor depth NPY exports.

Some OrcaLab builds write a 128-byte padded NPY header while declaring a
shorter header and big-endian float32.  The payload itself is native
little-endian float32.  NumPy consequently reads header padding as pixels and
byte-swaps the real depth values.  This loader validates the standard result
and falls back to the observed on-disk payload layout without modifying data.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _quality(depth: np.ndarray, near: float, far: float) -> float:
    return float((np.isfinite(depth) & (depth >= near) & (depth <= far)).mean())


def load_depth_npy(path: str | Path, shape=(480, 640), near=0.01, far=1024.0):
    """Return (metric_depth_float32, metadata), repairing the known bad header."""
    path = Path(path)
    standard = np.load(path).astype(np.float32)
    standard_quality = _quality(standard, near, far)
    if standard.shape == tuple(shape) and standard_quality >= 0.80:
        return standard, {"header_repaired": False, "valid_fraction": standard_quality}

    raw = path.read_bytes()
    count = int(np.prod(shape))
    payload_bytes = count * 4
    if len(raw) < payload_bytes:
        raise ValueError(f"Depth payload is truncated: {len(raw)} < {payload_bytes}")

    # The valid image occupies the final width*height float32 values.  Using
    # len(payload)-expected_size also tolerates future padding lengths.
    offset = len(raw) - payload_bytes
    repaired = np.frombuffer(raw, dtype="<f4", count=count, offset=offset).reshape(shape).copy()
    repaired_quality = _quality(repaired, near, far)
    if repaired_quality < 0.80:
        raise ValueError(
            f"Depth data is not metric after header repair: valid_fraction={repaired_quality:.3f}"
        )
    return repaired, {
        "header_repaired": True,
        "payload_offset": offset,
        "declared_dtype": str(np.load(path, mmap_mode="r").dtype),
        "actual_dtype": "<f4",
        "valid_fraction": repaired_quality,
    }
