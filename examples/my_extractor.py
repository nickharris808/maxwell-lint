"""Worked example: wrap your own extractor for maxwell-lint.

Run:
    PYTHONPATH=../src:. maxwell-lint check --extractor my_extractor:coupling
"""
from maxwell_lint.models import isolated_pair_matrix


def coupling(layout):
    """A deliberately naive extractor: pure 1/r pairwise, no screening at all.

    This is what pairwise superposition does -- it returns the isolated-pair
    value, i.e. k == 1 exactly. It sits precisely ON the ceiling, so it passes
    (k <= 1) while being maximally wrong in the sense of the error law: its
    screening depth is zero everywhere.

    Worth internalising before you read your own result: passing is cheap.
    """
    return isolated_pair_matrix(layout)


def coupling_broken(layout):
    """An extractor with a sign error in the screening term -> anti-screening."""
    return isolated_pair_matrix(layout) * 1.4
