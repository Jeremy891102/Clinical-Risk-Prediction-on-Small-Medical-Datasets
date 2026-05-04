"""Compute hyperparameter grid sizes for all models in MODEL_REGISTRY.

Output: results/grid_sizes.md (Markdown table) — used by the paper's
Methods/Experimental Setup section and the runtime discussion.
"""
from __future__ import annotations

import sys
from functools import reduce
from operator import mul
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.registry import MODEL_REGISTRY, get_model

OUTER_FOLDS = 10
INNER_FOLDS = 5


def grid_size(grid: dict) -> int:
    if not grid:
        return 0
    return reduce(mul, (len(v) for v in grid.values()), 1)


def fmt_params(grid: dict) -> str:
    if not grid:
        return "—"
    parts = []
    for k, v in grid.items():
        parts.append(f"`{k}`: {len(v)} values")
    return "; ".join(parts)


def main():
    rows = []
    for name, cls in MODEL_REGISTRY.items():
        try:
            model = get_model(name) if name != "tabpfn" else None
            if model is not None:
                grid = model.get_param_grid()
            else:
                grid = {}
        except Exception:
            grid = {}
        size = grid_size(grid)
        fits = 0 if size == 0 else size * INNER_FOLDS * OUTER_FOLDS
        rows.append((name, size, fits, grid, fmt_params(grid)))

    out = []
    out.append("# Hyperparameter Grid Sizes")
    out.append("")
    out.append(
        f"Nested CV: outer={OUTER_FOLDS}-fold × inner={INNER_FOLDS}-fold (GridSearchCV)."
    )
    out.append(
        "`Total fits` = grid size × inner folds × outer folds (per dataset)."
    )
    out.append("")
    out.append("| Model | Grid size | Total fits | Hyperparameters |")
    out.append("|---|---:|---:|---|")
    for name, size, fits, grid, params in rows:
        size_str = "0 (no tuning)" if size == 0 else str(size)
        fits_str = "—" if fits == 0 else str(fits)
        out.append(f"| {name} | {size_str} | {fits_str} | {params} |")
    out.append("")
    out.append("## Detailed grids")
    out.append("")
    for name, size, fits, grid, _ in rows:
        out.append(f"### {name}")
        if not grid:
            out.append("No hyperparameter tuning (foundation model uses default settings).")
            out.append("")
            continue
        for k, v in grid.items():
            out.append(f"- `{k}` ({len(v)}): `{v}`")
        out.append("")

    text = "\n".join(out)
    out_path = Path(__file__).resolve().parent.parent / "results" / "grid_sizes.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(text)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
