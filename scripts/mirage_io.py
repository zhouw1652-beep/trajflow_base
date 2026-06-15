"""
mirage_io.py -- Common I/O utilities for MIRAGE pkl files.

MIRAGE pkl files may be either:
1. Raw pickle files (legacy)
2. ZIP archives containing archive/data.pkl (newer format)

This module provides a single `load_mirage_pkl` function that handles both.

Usage:
    from scripts.mirage_io import load_mirage_pkl
    data = load_mirage_pkl("NewYork", TRAJFLOW_ROOT)

The returned dict has the standard MIRAGE keys:
    sequences, num_marks, num_seqs, num_pois, poi_gps, poi_category
"""

import io
import pickle
import zipfile
from pathlib import Path


def load_mirage_pkl(city, root):
    """
    Load a MIRAGE pkl file for a given city.

    Args:
        city: City name (e.g. "NewYork", "Tokyo", "Istanbul")
        root: TrajFlow root directory (pathlib.Path or str)

    Returns:
        A dict with keys: sequences, num_marks, num_seqs, num_pois, poi_gps, poi_category

    Raises:
        FileNotFoundError: If no MIRAGE pkl found for the city.
        pickle.UnpicklingError: If file cannot be unpickled.
    """
    root = Path(root)

    candidates = [
        root / "data" / "mirage_data" / f"mirage_{city.lower()}_processed.pkl",
        root / "data" / "mirage_data" / city / f"{city}.pkl",
    ]

    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                data = f.read()

            # Check magic bytes: ZIP files start with "PK"
            if data[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(data), "r") as z:
                    # Find archive/data.pkl inside the zip
                    members = z.namelist()
                    target = None
                    for m in members:
                        if m.endswith("data.pkl"):
                            target = m
                            break
                    if target is None:
                        raise ValueError(
                            f"ZIP archive at {path} does not contain a data.pkl file. "
                            f"Contents: {members}"
                        )
                    with z.open(target) as zf:
                        return pickle.load(zf)
            else:
                # Raw pickle bytes -- use pickle.loads
                return pickle.loads(data)

    raise FileNotFoundError(
        f"MIRAGE pkl for city '{city}' not found. Checked: {[str(p) for p in candidates]}"
    )
