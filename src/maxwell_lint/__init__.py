"""maxwell-lint -- does your coupling extractor predict physics that cannot exist?

Many-body screening can only *reduce* the coupling between two conductors below
their isolated-pair value. A predicted screening factor k > 1 is anti-screening,
and no passive arrangement of conductors in a linear medium can produce it.

This package tests any extractor against that ceiling.
"""
__version__ = "0.1.0"

from .ceiling import (  # noqa: F401
    CeilingReport, check_ceiling, pairwise_error, screening_depth, screening_factor,
)
from .models import (  # noqa: F401
    Layout, born_second_order, isolated_pair_matrix, mean_field,
    monopole_closure, random_layout,
)

__all__ = [
    "__version__", "CeilingReport", "check_ceiling", "pairwise_error",
    "screening_depth", "screening_factor", "Layout", "born_second_order",
    "isolated_pair_matrix", "mean_field", "monopole_closure", "random_layout",
]
