"""maxwell-lint CLI.

Two modes:

  maxwell-lint demo                    run the built-in models over a sweep
  maxwell-lint check --extractor M:F   test YOUR extractor against the ceiling

The adapter contract is deliberately tiny. Point `--extractor` at a dotted path
`module:function` where the function takes a Layout and returns an (N, N)
coupling matrix. If your extractor lives behind a file format or an API, wrap
it in five lines and point at the wrapper.

Exit codes:
  0  no ceiling violations
  1  ceiling violated -- the extractor predicts impossible physics
  2  usage / import error
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .ceiling import check_ceiling
from .models import (
    Layout,
    born_second_order,
    isolated_pair_matrix,
    mean_field,
    monopole_closure,
    random_layout,
)

BUILTIN = {
    "closure": monopole_closure,
    "born2": born_second_order,
    "mean_field": mean_field,
}

RED, GREEN, BOLD, DIM, RESET = "\033[31m", "\033[32m", "\033[1m", "\033[2m", "\033[0m"


def _c(t: str, col: str, use: bool) -> str:
    return f"{col}{t}{RESET}" if use else t


def _load_extractor(spec: str):
    if ":" not in spec:
        raise ValueError(
            f"expected 'module:function', got {spec!r} -- e.g. "
            "--extractor myproject.extract:coupling_matrix"
        )
    mod_name, fn_name = spec.split(":", 1)
    # A console script puts its own directory on sys.path, not the working
    # directory, so `--extractor myextract:f` for a myextract.py sitting right
    # there would fail with a bare ImportError. Look in cwd too.
    if "" not in sys.path and str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        raise ValueError(
            f"cannot import {mod_name!r}: {exc}. It must be importable from "
            f"{Path.cwd()} or installed in this environment -- check the name, "
            "or run from the directory that contains it."
        ) from exc
    fn = getattr(mod, fn_name, None)
    if fn is None:
        near = [a for a in dir(mod) if not a.startswith("_") and callable(getattr(mod, a))]
        hint = f" It does define: {', '.join(near[:6])}." if near else ""
        raise ValueError(f"{mod_name} has no attribute {fn_name!r}.{hint}")
    if not callable(fn):
        raise ValueError(f"{mod_name}.{fn_name} is not callable (it is a {type(fn).__name__})")
    return fn


def _read_matrix(path: str) -> np.ndarray:
    """Read a square matrix from .npy or a delimited text file."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"no such file: {p}")
    if p.suffix.lower() == ".npy":
        m = np.load(p)
    else:
        try:
            m = np.loadtxt(p, delimiter="," if "," in p.read_text(
                encoding="utf-8", errors="replace") else None)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"{p.name}: could not read as a numeric matrix ({exc}). "
                "Expected .npy, or text with one row per line and values "
                "separated by commas or whitespace."
            ) from exc
    m = np.asarray(m, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(
            f"{p.name}: expected a square (N, N) matrix, got shape {m.shape}. "
            "Each row is one conductor's coupling to every conductor."
        )
    if not np.all(np.isfinite(m)):
        n_bad = int(np.count_nonzero(~np.isfinite(m)))
        raise ValueError(
            f"{p.name}: {n_bad} non-finite value(s) (NaN/Inf). Refusing to "
            "check -- a ceiling verdict computed on NaN is not a verdict."
        )
    return m


def _read_layout(path: str, eps_r: float) -> "Layout":
    """Read conductor geometry: one row of `x_um, y_um, radius_um` per conductor."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"no such file: {p}")
    try:
        raw = np.loadtxt(p, delimiter="," if "," in p.read_text(
            encoding="utf-8", errors="replace") else None, ndmin=2)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{p.name}: could not read as numbers ({exc})") from exc
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(
            f"{p.name}: expected 3 columns (x_um, y_um, radius_um), got shape "
            f"{raw.shape}. One row per conductor."
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{p.name}: contains non-finite values")
    if np.any(raw[:, 2] <= 0):
        raise ValueError(f"{p.name}: every radius must be positive")
    return Layout(raw[:, :2] * 1e-6, raw[:, 2] * 1e-6, eps_r=eps_r)


def _run(fn, sizes, pitches, seed, diameter, tol):
    rows = []
    for n in sizes:
        for pitch in pitches:
            lay = random_layout(n, seed=seed, pitch_um=pitch, diameter_um=diameter)
            iso = isolated_pair_matrix(lay)
            rep = check_ceiling(np.asarray(fn(lay), dtype=float), iso, tol=tol)
            rows.append({"n": n, "pitch_um": pitch, **rep.as_dict()})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="maxwell-lint",
        description="Does your coupling extractor predict physics that cannot exist?",
    )
    p.add_argument("mode", choices=["demo", "check", "matrix"],
                   help="demo = built-in models; check = your extractor; "
                        "matrix = a coupling matrix you already have")
    p.add_argument("--extractor", help="dotted path module:function (check mode)")
    p.add_argument("--full", metavar="FILE",
                   help="matrix mode: your extractor's (N, N) coupling matrix "
                        "(.npy, or text with commas/whitespace)")
    p.add_argument("--isolated", metavar="FILE",
                   help="matrix mode: the isolated-pair reference matrix, same shape")
    p.add_argument("--layout", metavar="FILE",
                   help="matrix mode: conductor geometry, rows of x_um,y_um,radius_um; "
                        "the isolated-pair reference is computed from it")
    p.add_argument("--eps-r", type=float, default=4.6,
                   help="relative permittivity used with --layout (default 4.6)")
    p.add_argument("--sizes", default="6,8,12", help="conductor counts, comma-separated")
    p.add_argument("--pitches", default="60,80,100", help="pitches in um, comma-separated")
    p.add_argument("--diameter", type=float, default=40.0, help="conductor diameter in um")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-colour", action="store_true")
    p.add_argument("--version", action="version", version=f"maxwell-lint {__version__}")
    args = p.parse_args(argv)

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    pitches = [float(x) for x in args.pitches.split(",") if x.strip()]
    use = not args.no_colour and sys.stdout.isatty()

    if args.mode == "demo":
        out = {}
        for name, fn in BUILTIN.items():
            out[name] = _run(fn, sizes, pitches, args.seed, args.diameter, args.tol)
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(_c("maxwell-lint demo", BOLD, use))
            print(_c("  screening ceiling: k = |C_full|/|C_iso| <= 1 for all pairs", DIM, use))
            print()
            for name, rows in out.items():
                worst = max(r["max_k"] for r in rows)
                nv = sum(r["n_violations"] for r in rows)
                npair = sum(r["n_pairs"] for r in rows)
                tag = _c("PASS", GREEN, use) if nv == 0 else _c("FAIL", RED, use)
                pct = 100.0 * nv / npair if npair else 0.0
                print(f"  [{tag}] {name:<12} {nv:4d}/{npair:4d} pairs violate "
                      f"({pct:5.1f}%)   max k = {worst:.4f}")
            print()
            print(_c("  A predicted k > 1 is anti-screening: adding a grounded conductor", DIM, use))
            print(_c("  between two others cannot increase their coupling.", DIM, use))
        any_bad = any(r["n_violations"] for rows in out.values() for r in rows)
        return 1 if any_bad else 0

    if args.mode == "matrix":
        if not args.full:
            p.error("matrix mode requires --full FILE (your coupling matrix)")
        # The ceiling is a ratio, so it needs a reference. Without one there is
        # no verdict to give -- refuse rather than invent a comparison.
        if not (args.isolated or args.layout):
            p.error(
                "matrix mode needs a reference for the ratio k = |C_full|/|C_iso|: "
                "pass --isolated FILE if your tool produced the isolated-pair "
                "matrix, or --layout FILE (x_um,y_um,radius_um per conductor) to "
                "compute it here. There is no verdict without one."
            )
        if args.isolated and args.layout:
            p.error("pass either --isolated or --layout, not both")
        try:
            full = _read_matrix(args.full)
            if args.isolated:
                iso = _read_matrix(args.isolated)
                source = f"--isolated {args.isolated}"
            else:
                lay = _read_layout(args.layout, args.eps_r)
                if lay.n != full.shape[0]:
                    raise ValueError(
                        f"--layout has {lay.n} conductor(s) but --full is "
                        f"{full.shape[0]}x{full.shape[0]}; they must agree"
                    )
                iso = isolated_pair_matrix(lay)
                source = f"--layout {args.layout} (eps_r={args.eps_r:g})"
            if full.shape != iso.shape:
                raise ValueError(
                    f"shape mismatch: --full is {full.shape}, reference is "
                    f"{iso.shape}. Both must be (N, N) over the same conductors."
                )
            rep = check_ceiling(full, iso, tol=args.tol)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        if args.json:
            print(json.dumps({"full": args.full, "reference": source,
                              **rep.as_dict()}, indent=2))
        else:
            tag = _c("PASS", GREEN, use) if rep.passed else _c("FAIL", RED, use)
            print(f"[{tag}] {args.full}: {rep.n_violations}/{rep.n_pairs} pairs "
                  f"violate the ceiling, max k = {rep.max_k:.4f}")
            print(_c(f"       reference: {source}", DIM, use))
            if rep.worst_pair is not None and not rep.passed:
                i, j = rep.worst_pair
                print(_c(f"       worst pair: ({i}, {j})", DIM, use))
                print(_c("       A predicted k > 1 is anti-screening, which no passive "
                         "arrangement of conductors can produce.", DIM, use))
        return 1 if rep.n_violations else 0

    if not args.extractor:
        p.error("check mode requires --extractor module:function")
    try:
        fn = _load_extractor(args.extractor)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows = _run(fn, sizes, pitches, args.seed, args.diameter, args.tol)
    nv = sum(r["n_violations"] for r in rows)
    npair = sum(r["n_pairs"] for r in rows)
    worst = max(r["max_k"] for r in rows)

    if args.json:
        print(json.dumps({"extractor": args.extractor, "rows": rows,
                          "total_violations": nv, "total_pairs": npair,
                          "max_k": worst, "passed": nv == 0}, indent=2))
    else:
        tag = _c("PASS", GREEN, use) if nv == 0 else _c("FAIL", RED, use)
        print(f"[{tag}] {args.extractor}: {nv}/{npair} pairs violate the ceiling, "
              f"max k = {worst:.4f}")
        if nv:
            print(_c("      This extractor predicts anti-screening, which no passive "
                     "arrangement of conductors can produce.", DIM, use))
    return 1 if nv else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
