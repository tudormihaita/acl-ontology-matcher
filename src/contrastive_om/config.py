"""Experiment configuration: YAML files with single inheritance (``inherit: base.yaml``)."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class Config:
    """Nested, read-only-ish view over a config dict with attribute access.

    ``cfg.pairs.positives`` and ``cfg["pairs"]["positives"]`` are equivalent.
    """

    def __init__(self, data: dict[str, Any], path: Path | None = None):
        self._data = data
        self._path = path

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(f"config has no key '{key}'") from exc

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value) if isinstance(value, dict) else value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self._data else default

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    @property
    def path(self) -> Path | None:
        return self._path

    def hash(self, *sections: str) -> str:
        """Short stable hash of the given top-level sections (all if none given)."""
        keys = sections or tuple(sorted(self._data))
        payload = {k: self._data.get(k) for k in keys}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_raw(path: Path) -> dict:
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    parent = data.pop("inherit", None)
    if parent:
        data = _deep_merge(_load_raw((path.parent / parent).resolve()), data)
    return data


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    path = Path(path)
    data = _load_raw(path)
    if overrides:
        data = _deep_merge(data, overrides)
    return Config(data, path)


def resolve_device(name: str = "auto") -> str:
    """``auto`` -> cuda > mps > cpu; otherwise the name itself."""
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (if installed) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass
