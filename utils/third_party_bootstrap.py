from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_IMPORT_PATHS = {
    "depth_anything_3": [PROJECT_ROOT / "third_party" / "depth-anything-3" / "src"],
    "romav2": [PROJECT_ROOT / "third_party" / "RoMaV2" / "src"],
    "torch_kdtree": [PROJECT_ROOT / "third_party" / "torch_kdtree"],
}


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def prepend_local_third_party_paths(*package_names: str) -> list[Path]:
    added_paths: list[Path] = []
    existing_paths = {_normalized_path(Path(entry)) for entry in sys.path if entry}

    for package_name in package_names:
        for path in THIRD_PARTY_IMPORT_PATHS.get(package_name, []):
            if not path.exists():
                continue
            normalized = _normalized_path(path)
            if normalized in existing_paths:
                continue
            sys.path.insert(0, str(path))
            existing_paths.add(normalized)
            added_paths.append(path)

    return added_paths
