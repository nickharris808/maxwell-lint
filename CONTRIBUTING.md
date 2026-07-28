# Contributing to maxwell-lint

## The one rule

If you add a model, add it to **both** directions of the test suite: a model
that should respect the ceiling needs a test asserting it passes, and a model
that should violate it needs a test asserting it fails.

A ceiling test that only ever passes is indistinguishable from one that is not
running at all. The `born2` model exists precisely so the suite has something
that must go red.

## Running the tests

```bash
pip install -e ".[dev]"
pytest -q
maxwell-lint demo          # the worked example from the README
```

## Adding an extractor adapter

Adapters do not belong in this package. Write a five-line wrapper in your own
code exposing `f(layout) -> (N, N) ndarray` and point `--extractor` at it. The
adapter contract is intentionally tiny so it cannot drift.

## Scope

This package tests a **necessary condition**. Please do not add accuracy
metrics, reference solvers, or fitted corrections — they belong in the tool
being graded, not in the grader. Keeping the grader independent of any
particular extractor is what makes its verdict worth having.

numpy is the only runtime dependency. Please keep it that way.
