"""Write the matrix-mode example files: geometry, and two coupling matrices.

Deterministic. `born2.csv` is the truncated-perturbation model, which predicts
impossible physics on this array; `closure.csv` is the global-inversion model,
which does not. Regenerate with:

    python examples/make_matrix_examples.py
"""
from __future__ import annotations

import pathlib

import numpy as np

from maxwell_lint.models import (
    born_second_order,
    isolated_pair_matrix,
    monopole_closure,
    random_layout,
)

HERE = pathlib.Path(__file__).resolve().parent


def main() -> None:
    lay = random_layout(8, seed=1, pitch_um=60.0, diameter_um=40.0)
    out = {
        "geometry.csv": np.column_stack([lay.xy * 1e6, lay.radius * 1e6]),
        "closure.csv": monopole_closure(lay),
        "born2.csv": born_second_order(lay),
        "isolated.csv": isolated_pair_matrix(lay),
    }
    for name, arr in out.items():
        np.savetxt(HERE / name, arr, delimiter=",", fmt="%.12e", newline="\n")
        print(f"wrote {name}  {arr.shape}")


if __name__ == "__main__":
    main()
