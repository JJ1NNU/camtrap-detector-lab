"""Generic name->class registry used by the factory (no hardcoded if/else)."""
from __future__ import annotations
from typing import Any, Callable, Dict

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._store: Dict[str, Callable[..., Any]] = {}

    def register(self, key: str):
        def deco(obj):
            if key in self._store:
                raise KeyError(f"'{key}' already registered in '{self.name}'")
            self._store[key] = obj
            return obj
        return deco

    def get(self, key: str):
        if key not in self._store:
            raise KeyError(f"'{key}' not found in registry '{self.name}'. "
                           f"Available: {list(self._store)}")
        return self._store[key]

    def build(self, cfg: dict):
        """cfg is a dict with a 'name' key + constructor kwargs."""
        cfg = dict(cfg)
        key = cfg.pop("name")
        return self.get(key)(**cfg)

    def available(self):
        return list(self._store)

DETECTORS = Registry("detectors")
STRATEGIES = Registry("strategies")
