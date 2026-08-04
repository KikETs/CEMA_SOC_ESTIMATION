from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path


def load_package(path):
    entry = Path(path).resolve()
    source = entry / "code" / "inference.py"
    if not source.is_file():
        raise FileNotFoundError(f"Frozen inference code not found: {source}")
    name = "_t6_plain_frozen_" + hashlib.sha256(str(entry).encode("utf-8")).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen inference module: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.load_package(entry)
