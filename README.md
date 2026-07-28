# maxwell-lint

![CI](https://github.com/nickharris808/maxwell-lint/actions/workflows/ci.yml/badge.svg) ![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.13-blue) ![Licence](https://img.shields.io/badge/licence-Apache--2.0-green) ![Tests](https://img.shields.io/badge/tests-27%20passing-brightgreen)

**Does your coupling extractor predict physics that cannot exist?**

## Why this exists

When conductors are packed together, every other conductor *screens* the field
between any two. So the mutual capacitance of a pair inside an array is always
**at most** its isolated-pair value:

```
k = |C_full| / |C_iso|  ≤  1
```

A predicted `k > 1` is **anti-screening** — it says that adding a grounded
conductor between two others *increases* their coupling. No passive arrangement
of conductors in a linear medium can do that. If your extractor predicts it,
your extractor is wrong, and no accuracy metric will tell you.

```bash
# from source (works today)
git clone <this-repo> && pip install ./maxwell-lint

maxwell-lint check --extractor mypackage.extract:coupling_matrix
```

> **Not yet on PyPI.** `pip install maxwell-lint` is the intended install once published; until then use the source install above.

## 30-second quickstart

```bash
$ maxwell-lint demo
maxwell-lint demo
  screening ceiling: k = |C_full|/|C_iso| <= 1 for all pairs

  [PASS] closure         0/ 654 pairs violate (  0.0%)   max k = 0.8283
  [FAIL] born2         510/ 654 pairs violate ( 78.0%)   max k = 3.4658
  [PASS] mean_field      0/ 654 pairs violate (  0.0%)   max k = 0.4490

  A predicted k > 1 is anti-screening: adding a grounded conductor
  between two others cannot increase their coupling.
```

Reproduce exactly: `maxwell-lint demo --sizes 6,8,12 --pitches 60,80,100 --json`
(baseline committed at [`examples/baseline.json`](examples/baseline.json)).

## What that result means

`born2` is a **second-order Born approximation** — a truncated perturbation
series, and the thing a competent engineer reaches for when a full solve is too
slow. It is not a strawman; it is the obvious cheap correction.

Truncating an alternating series overshoots, and here the overshoot pushes the
predicted screening factor **above 1 on 78% of pairs**, peaking at `k = 3.47`.
A first-order correction to a known structural error makes it *worse* than the
uncorrected value in a way that is qualitatively unphysical.

`closure` is a global inversion — it captures all scattering orders — and it
never violates the ceiling. `mean_field` also respects the ceiling: it is crude
but not *impossible*, which is the distinction this tool draws.

## Test your own extractor

The adapter contract is one function:

```python
# mypackage/extract.py
def coupling_matrix(layout):
    """layout.xy is (N,2) metres, layout.radius is (N,) metres.
    Return an (N,N) coupling matrix."""
    ...
```

```bash
maxwell-lint check --extractor mypackage.extract:coupling_matrix \
                   --sizes 6,8,12 --pitches 60,80,100
```

Exit codes: `0` no violations · `1` ceiling violated · `2` usage/import error.

## Library use

```python
from maxwell_lint import check_ceiling, isolated_pair_matrix, random_layout

lay = random_layout(8, seed=1, pitch_um=80.0)
report = check_ceiling(my_extractor(lay), isolated_pair_matrix(lay))
print(report.summary())
```

## The error law that follows

Writing the **screening depth** as `δ = −log₁₀ k ≥ 0`, a pairwise-superposition
extractor (which assumes `k ≡ 1`) has relative error

```
E(δ) = 10^δ − 1
```

`maxwell_lint.pairwise_error` implements it. Three properties, and each is a
statement about *every* depth rather than a fitted trend:

- **E(0) = 0** — pairwise is exact if and only if there is no screening;
- **E(δ) ≥ 0** — the error is one-sided; pairwise never *under*-predicts;
- **E is strictly increasing** — there is no depth beyond which it stops getting
  worse, so a pairwise extractor becomes more wrong as designs densify.

## Scope, honestly

This is a **necessary-condition** test. It tells you an extractor is wrong; it
cannot tell you one is right. An extractor that returns `k = 0.5` everywhere
passes the ceiling and is useless.

The reference models here are **thin-wire / monopole** forms in a homogeneous
medium — good enough to demonstrate the ceiling and to sanity-check your own
extractor's sign and magnitude, not a substitute for a field solve. They are
2-D per-unit-length; finite-length end effects are out of scope.

## Where the accurate extractor lives

`maxwell-lint` grades extractors. Building one that is both fast and *correct*
in the many-body regime is the [ChipletOS](https://chipletos.com) closed core —
a learned many-body coupling operator trained on an owned solver corpus, with
calibrated abstention and a fail-closed signoff certificate.

## License

Apache-2.0. See [LICENSE](LICENSE); copyright is declared in [NOTICE](NOTICE).
