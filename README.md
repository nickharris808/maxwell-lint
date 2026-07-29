# maxwell-lint

![CI](https://github.com/nickharris808/maxwell-lint/actions/workflows/ci.yml/badge.svg) ![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.13-blue) ![Licence](https://img.shields.io/badge/licence-Apache--2.0-green) ![Tests](https://img.shields.io/badge/tests-55%20passing-brightgreen)

📖 **[Documentation site](https://nickharris808.github.io/physics-lint/)** — the portfolio narrative, the concepts, a full walkthrough, and what all of this proves (and does not).

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
pip install git+https://github.com/nickharris808/maxwell-lint.git

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

## Or check a matrix you already have

Writing an adapter is fine when your extractor is a Python function. Often it
is not — it is a licensed tool that wrote a file. `matrix` mode takes the
numbers directly:

```bash
maxwell-lint matrix --full coupling.csv --layout geometry.csv
```

`--full` is your extractor's (N, N) coupling matrix, as `.npy` or as text with
values separated by commas or whitespace. `--layout` is the geometry it came
from — one row of `x_um, y_um, radius_um` per conductor — from which the
isolated-pair reference is computed here.

If your tool already produced the isolated-pair matrix, hand that over instead
and no geometry is needed:

```bash
maxwell-lint matrix --full coupling.csv --isolated isolated.csv
```

**With neither, the command refuses.** `k = |C_full| / |C_iso|` is a ratio, and
there is no verdict to give without the denominator:

```
maxwell-lint: error: matrix mode needs a reference for the ratio
k = |C_full|/|C_iso|: pass --isolated FILE if your tool produced the
isolated-pair matrix, or --layout FILE (x_um,y_um,radius_um per conductor) to
compute it here. There is no verdict without one.
```

A worked run you can reproduce — `examples/` ships an 8-conductor array and two
matrices for it, one from each reference model, standing in for your tool:

```bash
$ maxwell-lint matrix --full examples/born2.csv --layout examples/geometry.csv
[FAIL] examples/born2.csv: 28/56 pairs violate the ceiling, max k = 1.5946
       reference: --layout examples/geometry.csv (eps_r=4.6)
       worst pair: (2, 6)
       A predicted k > 1 is anti-screening, which no passive arrangement of conductors can produce.

$ maxwell-lint matrix --full examples/closure.csv --layout examples/geometry.csv
[PASS] examples/closure.csv: 0/56 pairs violate the ceiling, max k = 0.8062
       reference: --layout examples/geometry.csv (eps_r=4.6)
```

Same geometry, same ceiling, opposite verdicts — the difference is entirely in
the model that produced the matrix. Conductor 2 and conductor 6 are the pair to
look at first.

Both reference paths agree exactly: `--isolated examples/isolated.csv` on the
same array gives the same 28/56 and the same `max k`, because the only thing
`--layout` does is compute that matrix for you. The files are regenerated by
`python examples/make_matrix_examples.py`.

## Or start from an S-parameter file

If what you have is a measured or simulated N-port rather than a capacitance
matrix, `--from-touchstone` derives a coupling proxy from it (needs
[`touchstone-tools`](https://github.com/nickharris808/touchstone-tools)):

```bash
maxwell-lint matrix --from-touchstone array.s4p --isolated isolated_pairs.csv
```

**Read this before using it.** The proxy is `|S_ij|` at one frequency — the
transmitted magnitude between ports i and j, a monotone stand-in for how
strongly they couple. It is **not** a capacitance extraction: it carries the
network's loss, its reference plane and its port impedances along with the
coupling, so its absolute scale is not `C_ij`.

That has a hard consequence, and the tool enforces it rather than leaving it to
the reader: **`--from-touchstone` refuses to work with `--layout`.**

```
maxwell-lint: error: --from-touchstone yields |S|, a transmitted-magnitude
proxy, while --layout computes an isolated-pair *capacitance*. Those are
different quantities and their ratio means nothing. Use --isolated with an
isolated-pair S-matrix measured or simulated the same way.
```

The ratio `k = |C_full|/|C_iso|` only means something when numerator and
denominator are the same kind of quantity. Comparing an `|S|` proxy against a
capacitance reference would produce a number for every pair, and every one of
them would be meaningless.

`--freq-hz` picks the frequency; the default is the lowest in the file. A
frequency that is not in the file is **refused**, not snapped to the nearest
sample — snapping would answer a question you did not ask.

## Library use

```python
from maxwell_lint import check_ceiling, isolated_pair_matrix, random_layout

lay = random_layout(8, seed=1, pitch_um=80.0)
report = check_ceiling(my_extractor(lay), isolated_pair_matrix(lay))
print(report.summary())
```

| Object | What it is |
|---|---|
| `Layout(xy, radius, eps_r=4.6)` | Parallel circular conductors. `xy` is (N, 2) **metres**, `radius` is (N,) metres. `.n`, `.distances()` |
| `random_layout(n, seed, pitch_um, diameter_um, jitter)` | A manufacturable-ish array with min pitch respected. |
| `isolated_pair_matrix(layout)` | Each pair's mutual capacitance with only that pair present — the ceiling's denominator. |
| `monopole_closure(layout)` | Full-array coupling by global inversion. Respects the ceiling. |
| `born_second_order(layout)` | Truncated Neumann series. Predicts `k > 1`, on purpose. |
| `mean_field(layout)` | Averaged screening, positions unresolved. |
| `screening_factor(c_full, c_iso)` | Elementwise `k`. Diagonal is `NaN` — a self term is not a screened coupling. |
| `screening_depth(k)` | `δ = −log₁₀ k`. |
| `pairwise_error(δ)` | `10^δ − 1`, the relative error of assuming `k ≡ 1`. |
| `check_ceiling(c_full, c_iso, tol=1e-9) -> CeilingReport` | The verdict. |

`CeilingReport` carries `.n_pairs`, `.n_violations`, `.violation_fraction`,
`.max_k`, `.worst_pair`, `.median_depth`, `.median_pairwise_error`, `.passed`,
`.tol`, plus `.as_dict()` and `.summary()`.

## CLI reference

```
maxwell-lint {demo,check,matrix} [options]
```

| Mode | What it does |
|---|---|
| `demo` | Run the three built-in models over a sweep. Needs nothing from you. |
| `check` | Call **your** extractor over generated layouts. Needs `--extractor`. |
| `matrix` | Check numbers you already have. Needs `--full` plus a reference. |

| Option | Applies to | Meaning |
|---|---|---|
| `--extractor MODULE:FUNCTION` | `check` | Dotted path to a callable taking a `Layout`, returning (N, N). The working directory is searched as well as the install path. |
| `--full FILE` | `matrix` | Your (N, N) coupling matrix: `.npy`, or text with commas/whitespace. |
| `--isolated FILE` | `matrix` | The isolated-pair reference, same shape. |
| `--layout FILE` | `matrix` | Geometry as `x_um,y_um,radius_um` rows; the reference is computed from it. |
| `--from-touchstone FILE` | `matrix` | Derive a coupling proxy from an N-port S-matrix instead of `--full`. Requires `touchstone-tools`. Cannot be combined with `--layout`. |
| `--freq-hz F` | `matrix` | With `--from-touchstone`: which frequency. Default the lowest; a frequency not in the file is refused. |
| `--eps-r FLOAT` | `matrix` | Relative permittivity used with `--layout`. Default `4.6`. |
| `--sizes N,N,...` | `demo`, `check` | Conductor counts. Default `6,8,12`. |
| `--pitches U,U,...` | `demo`, `check` | Pitches in µm. Default `60,80,100`. |
| `--diameter U` | `demo`, `check` | Conductor diameter in µm. Default `40`. |
| `--seed N` | `demo`, `check` | Layout seed. Default `1`. |
| `--tol F` | all | Slack above `k = 1` before a pair counts as a violation. Default `1e-9`. |
| `--json` | all | Machine-readable output. |
| `--no-colour` | all | Disable ANSI colour. |
| `--version` | — | Print the version. |

Exit codes: `0` clean · `1` ceiling violated · `2` usage, import or input error.

## Troubleshooting

**`cannot import 'myextract'`** — a console script puts its own directory on
`sys.path`, not yours. `maxwell-lint` also searches the working directory, so
run it from the directory containing the module, or install the package.

**`mypackage.extract has no attribute 'coupling'`** — the message lists what the
module does define. Usually a typo or an inner function that was never exported.

**`expected a square (N, N) matrix, got shape (8, 3)`** — you passed the layout
to `--full`. Geometry has three columns; the coupling matrix is square.

**`--layout has 8 conductor(s) but --full is 12x12`** — the matrix and the
geometry describe different arrays. Check you exported both from the same run.

**`2 non-finite value(s) (NaN/Inf). Refusing to check`** — deliberate. A ceiling
verdict computed on `NaN` is not a verdict, and there is no override.

**Every pair reports `k` far below 1 and the run passes trivially** — check the
units. `Layout` takes **metres**; `--layout` files take **micrometres**. Mixing
them scales every distance by 10⁶ and every conductor looks isolated.

**`--from-touchstone` and `--layout` together are refused** — deliberately. See
above: `|S|` and capacitance are different quantities and their ratio means
nothing.

**`no sample near 3 GHz; the closest is 1 GHz`** — pick a frequency that is
actually in the file. Snapping silently would give you a verdict about a
different frequency than the one you asked about.

**`demo` exits 1 and you expected 0** — that is correct. The demo deliberately
includes `born2`, a model that must fail; an exit code of 0 there would mean the
checker had stopped discriminating.

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

## The rest of the toolkit

Eight artifacts that answer one question in different places: **is this
model physically possible?** Each is a grader — it can tell you a model is
wrong; none can tell you one is right.

| | |
|---|---|
| [`sparam-lint`](https://github.com/nickharris808/sparam-lint) | Is an S-parameter model physically possible? Five laws + a negative control. |
| [`maxwell-lint`](https://github.com/nickharris808/maxwell-lint) ← you are here | Does a coupling extractor predict impossible physics? Screening ceiling k ≤ 1. |
| [`abstain-bench`](https://github.com/nickharris808/abstain-bench) | Does a model know when to shut up? Abstention recall, never pooled with accuracy. |
| [`sparam-conformance`](https://huggingface.co/datasets/nickh007/sparam-conformance) | 11 labelled networks with verified ground truth. Grades the graders. |
| [`screening-ceiling`](https://huggingface.co/datasets/nickh007/screening-ceiling) | A certified impossibility result + 27 counterexamples. Zero-dependency verifier. |
| [`physics-lint-action`](https://github.com/nickharris808/physics-lint-action) | The same checks, in your CI. |
| [`physics-lint-mcp`](https://github.com/nickharris808/physics-lint-mcp) | A physics oracle your AI agent can call. |
| [**Try it in your browser**](https://huggingface.co/spaces/nickh007/physics-lint) | All three checks, no install, runs client-side. |

These tools **grade** a model. Producing one that is passive *by
construction* — so it cannot fail these laws whatever its parameters — and
accurate at speed in the many-body regime, with calibrated abstention and a
fail-closed signoff certificate, is the commercial core:
**[ChipletOS](https://chipletos.com)**.

## Licence

Apache-2.0. See [LICENSE](LICENSE); copyright is declared in [NOTICE](NOTICE).

## Contributing

One non-negotiable rule here: a new reference model must be one a competent engineer would actually reach for, not a strawman built to lose. [`CONTRIBUTING.md`](CONTRIBUTING.md) has the detail. Each sibling repository states its own, and they differ — that is deliberate, and it is why each is trustworthy on its own terms.

## Citation

[`CITATION.cff`](CITATION.cff) is machine-readable; GitHub renders a “Cite this repository” button from it.
