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

import numpy as np

from . import __version__
from .ceiling import check_ceiling
from .models import (
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
        raise ValueError(f"expected 'module:function', got {spec!r}")
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"{mod_name} has no callable {fn_name!r}")
    return fn


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
    p.add_argument("mode", choices=["demo", "check"], help="demo = built-in models")
    p.add_argument("--extractor", help="dotted path module:function (check mode)")
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
