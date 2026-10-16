"""Locate and load bundled example datasets from the examples folder."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from config.settings import PathManager
from . import load_inputs

REQUIRED_KEYS = (
    "customers",
    "source",
    "lv_conductors",
    "mv_conductors",
    "transformers",
    "standalone_packages",
    "cost_parameters",
)

FILE_CANDIDATES: Dict[str, tuple[str, ...]] = {
    "customers": ("customers.csv", "customers_example.csv"),
    "source": ("source.csv", "source_example.csv"),
    "lv_conductors": ("lv_conductors.csv", "lv_conductors_example.csv"),
    "mv_conductors": ("mv_conductors.csv", "mv_conductors_example.csv"),
    "transformers": ("transformers.csv", "transformers_example.csv"),
    "standalone_packages": ("standalone_packages.csv", "standalone_packages_example.csv"),
    "cost_parameters": (
        "cost_parameters.csv",
        "cost_parameters_example.csv",
        "cost_parameters.yaml",
        "cost_parameters.yml",
    ),
    "roads": ("roads.geojson", "roads_example.geojson", "roads.gpkg", "roads_example.gpkg"),
}

LEGACY_ALIASES = {
    "synthetic": "mvp",
    "mafa": "mafa case study",
    "realistic_ssa": "complete",
}

DATASET_LABEL_OVERRIDES = {
    "mvp": "mvp - Synthetic demo",
    "nga_1": "NGA 1 compact village",
    "nga_dense": "NGA-dense compact village (Mafa counterfactual)",
    "mafa case study": "Mafa case study, NGA",
    "complete": "complete - Realistic SSA village",
}

_LOADERS = {
    "customers": load_inputs.load_customers,
    "source": load_inputs.load_source,
    "lv_conductors": load_inputs.load_lv_conductors,
    "mv_conductors": load_inputs.load_mv_conductors,
    "transformers": load_inputs.load_transformers,
    "standalone_packages": load_inputs.load_standalone_packages,
    "cost_parameters": load_inputs.load_cost_parameters,
}


def _find_file(base: Path, candidates: tuple[str, ...]) -> str | None:
    existing = {p.name.casefold(): p.name for p in base.iterdir() if p.is_file()}
    for candidate in candidates:
        match = existing.get(candidate.casefold())
        if match is not None:
            return match
    return None


def _count_csv_rows(path: Path) -> int | None:
    if path.suffix.casefold() != ".csv" or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except OSError:
        return None


def _project_label(project_name: str, files: Dict[str, str], base: Path) -> str:
    label = DATASET_LABEL_OVERRIDES.get(project_name, project_name)
    rows = _count_csv_rows(base / files["customers"])
    return f"{label} ({rows:,} customers)" if rows is not None else label


def _project_spec(base: Path) -> Dict[str, Any] | None:
    files: Dict[str, str] = {}
    for key, candidates in FILE_CANDIDATES.items():
        match = _find_file(base, candidates)
        if match is not None:
            files[key] = match
    if any(key not in files for key in REQUIRED_KEYS):
        return None
    return {
        "label": _project_label(base.name, files, base),
        "project_name": base.name,
        "dir": base,
        "files": files,
    }


def discover_example_datasets() -> Dict[str, Dict[str, Any]]:
    """Discover loadable MVP project folders directly under examples/."""
    root = PathManager.EXAMPLES
    if not root.exists():
        return {}
    datasets: Dict[str, Dict[str, Any]] = {}
    for folder in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.casefold()):
        spec = _project_spec(folder)
        if spec is not None:
            datasets[folder.name] = spec
    return datasets


def _with_legacy_aliases(datasets: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = dict(datasets)
    for alias, target in LEGACY_ALIASES.items():
        if target in datasets:
            out[alias] = {**datasets[target], "alias_for": target}
    return out


def refresh_dataset_registry() -> Dict[str, Dict[str, Any]]:
    """Refresh the compatibility registry, including legacy aliases."""
    return _with_legacy_aliases(discover_example_datasets())


# Registry of selectable example datasets, discovered from examples/.
DATASETS: Dict[str, Dict[str, Any]] = refresh_dataset_registry()

DEFAULT_DATASET = "mvp" if "mvp" in DATASETS else next(iter(DATASETS), "")


def _resolve_dataset_key(dataset: str) -> str:
    key = LEGACY_ALIASES.get(str(dataset), str(dataset))
    if key not in DATASETS:
        raise KeyError(f"Unknown example dataset/project: {dataset}")
    return key


def load_example_inputs(dataset: str = DEFAULT_DATASET) -> Dict[str, Any]:
    """Return normalized MVP inputs for a discovered example project."""
    global DATASETS
    DATASETS = refresh_dataset_registry()
    dataset = _resolve_dataset_key(dataset)
    spec = DATASETS[dataset]
    base: Path = spec["dir"]
    out: Dict[str, Any] = {key: loader(base / spec["files"][key]) for key, loader in _LOADERS.items()}
    roads_file = spec["files"].get("roads")
    out["roads"] = load_inputs.load_roads(base / roads_file) if roads_file else None
    return out
