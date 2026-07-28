"""Test suite for maxwell-lint.

The load-bearing tests are the two directions: a physically-correct model must
PASS the ceiling, and a known-broken one must FAIL it. A ceiling test that only
ever passes is indistinguishable from one that is not running.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maxwell_lint import (  # noqa: E402
    born_second_order,
    check_ceiling,
    isolated_pair_matrix,
    mean_field,
    monopole_closure,
    pairwise_error,
    random_layout,
    screening_depth,
    screening_factor,
)
from maxwell_lint.cli import main as cli_main  # noqa: E402
from maxwell_lint.models import Layout  # noqa: E402

# ------------------------------------------------------------------ the ceiling

@pytest.mark.parametrize("n,pitch", [(6, 60.0), (8, 80.0), (8, 100.0), (12, 60.0)])
def test_global_closure_respects_ceiling(n, pitch):
    """A global inversion captures all scattering orders and cannot anti-screen."""
    lay = random_layout(n, seed=1, pitch_um=pitch)
    rep = check_ceiling(monopole_closure(lay), isolated_pair_matrix(lay))
    assert rep.passed, rep.summary()
    assert rep.n_violations == 0
    assert rep.max_k <= 1.0


@pytest.mark.parametrize("n,pitch", [(8, 100.0), (12, 60.0)])
def test_second_order_born_violates_ceiling(n, pitch):
    """The load-bearing negative: a truncated series overshoots past k = 1."""
    lay = random_layout(n, seed=1, pitch_um=pitch)
    rep = check_ceiling(born_second_order(lay), isolated_pair_matrix(lay))
    assert not rep.passed, "Born-2 should violate the ceiling; the check is not discriminating"
    assert rep.n_violations > 0
    assert rep.max_k > 1.0


def test_ceiling_detects_a_hand_built_violation():
    """Independent of any model: hand a k > 1 matrix straight to the checker."""
    iso = np.array([[0.0, 1.0], [1.0, 0.0]])
    full = np.array([[0.0, 1.5], [1.5, 0.0]])
    rep = check_ceiling(full, iso)
    assert not rep.passed
    assert rep.max_k == pytest.approx(1.5)
    assert rep.worst_pair in {(0, 1), (1, 0)}


def test_ceiling_accepts_a_hand_built_screened_pair():
    iso = np.array([[0.0, 1.0], [1.0, 0.0]])
    full = np.array([[0.0, 0.6], [0.6, 0.0]])
    rep = check_ceiling(full, iso)
    assert rep.passed
    assert rep.max_k == pytest.approx(0.6)


# ------------------------------------------------------------------- the algebra

def test_screening_factor_nulls_the_diagonal():
    iso = np.ones((3, 3))
    full = np.ones((3, 3)) * 0.5
    k = screening_factor(full, iso)
    assert np.all(np.isnan(np.diag(k)))
    assert np.allclose(k[~np.eye(3, dtype=bool)], 0.5)


def test_error_law_zero_at_zero_depth():
    assert pairwise_error(0.0) == pytest.approx(0.0)


def test_error_law_is_strictly_increasing():
    d = np.linspace(0.0, 2.0, 50)
    e = pairwise_error(d)
    assert np.all(np.diff(e) > 0), "E(delta) must be strictly increasing"


def test_error_law_is_nonnegative_for_nonnegative_depth():
    d = np.linspace(0.0, 3.0, 100)
    assert np.all(pairwise_error(d) >= 0.0)


def test_depth_and_error_are_consistent():
    """E(delta) must equal |1/k - 1| for the same k."""
    k = np.array([1.0, 0.5, 0.1, 0.01])
    d = screening_depth(k)
    assert np.allclose(pairwise_error(d), np.abs(1.0 / k - 1.0))


def test_shape_mismatch_refused():
    with pytest.raises(ValueError, match="shape mismatch"):
        screening_factor(np.ones((3, 3)), np.ones((4, 4)))


def test_non_square_refused():
    with pytest.raises(ValueError, match="square"):
        screening_factor(np.ones((2, 3)), np.ones((2, 3)))


def test_nonfinite_pairs_are_skipped_and_counted():
    iso = np.array([[0.0, 1.0], [1.0, 0.0]])
    full = np.array([[0.0, np.nan], [np.nan, 0.0]])
    rep = check_ceiling(full, iso)
    assert rep.n_pairs == 0
    assert not rep.passed
    assert "no finite" in rep.detail.get("reason", "")


# -------------------------------------------------------------------- the models

def test_layout_validates_shapes():
    with pytest.raises(ValueError, match=r"xy must be"):
        Layout(np.zeros((3,)), np.ones(3))
    with pytest.raises(ValueError, match=r"radius must be"):
        Layout(np.zeros((3, 2)), np.ones(4))


def test_random_layout_is_deterministic():
    a = random_layout(8, seed=7)
    b = random_layout(8, seed=7)
    assert np.array_equal(a.xy, b.xy)


def test_random_layout_respects_requested_count():
    for n in (3, 6, 9, 16):
        assert random_layout(n, seed=0).n == n


def test_mean_field_also_respects_ceiling_but_is_crude():
    """Mean-field does not anti-screen; it is simply inaccurate. Included so the
    ceiling test is not mistaken for an accuracy test."""
    lay = random_layout(8, seed=3, pitch_um=80.0)
    rep = check_ceiling(mean_field(lay), isolated_pair_matrix(lay))
    assert rep.passed


# ------------------------------------------------------------------------- cli

def test_cli_demo_flags_the_broken_model(capsys):
    rc = cli_main(["demo", "--sizes", "8", "--pitches", "100", "--no-colour"])
    out = capsys.readouterr().out
    assert rc == 1, "demo includes a violating model, so overall rc must be 1"
    assert "closure" in out and "born2" in out
    assert "PASS" in out and "FAIL" in out


def test_cli_demo_json_shape(capsys):
    rc = cli_main(["demo", "--sizes", "6", "--pitches", "80", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc in (0, 1)
    assert set(payload) == {"closure", "born2", "mean_field"}
    for rows in payload.values():
        assert rows and "max_k" in rows[0] and "n_violations" in rows[0]


def test_cli_check_with_user_extractor(capsys):
    """The bring-your-own-extractor adapter contract."""
    rc = cli_main(["check", "--extractor", "maxwell_lint.models:monopole_closure",
                   "--sizes", "8", "--pitches", "100", "--no-colour"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS" in out


def test_cli_check_catches_bad_extractor(capsys):
    rc = cli_main(["check", "--extractor", "maxwell_lint.models:born_second_order",
                   "--sizes", "8", "--pitches", "100", "--no-colour"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out and "anti-screening" in out


def test_cli_bad_extractor_spec_exits_two(capsys):
    assert cli_main(["check", "--extractor", "not_a_module_at_all:f"]) == 2
    assert cli_main(["check", "--extractor", "noconcolon"]) == 2


def test_module_entrypoint_runs():
    r = subprocess.run(
        [sys.executable, "-m", "maxwell_lint.cli", "demo",
         "--sizes", "6", "--pitches", "80", "--json"],
        capture_output=True, text=True,
        env={**os.environ,
             # Inherit the OS environment and override only PYTHONPATH.
             # A scrubbed env is not portable: on Windows, Python needs
             # SYSTEMROOT to seed its hash randomisation and aborts
             # without it.
             "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    assert r.returncode in (0, 1), r.stderr
    assert "closure" in json.loads(r.stdout)


# ----------------------------------------------------------- packaging boundary

def test_package_imports_nothing_private():
    root = Path(__file__).resolve().parents[1] / "src"
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for forbidden in ("import genesis", "from genesis", "provisionals"):
            assert forbidden not in text, f"{py.name} references private tree: {forbidden}"


# ------------------------------------------- every shipped install must work

_REPO = Path(__file__).resolve().parents[1]


def test_no_install_command_points_at_a_path_a_stranger_will_not_have():
    """READMEs get copy-pasted, so a local path in one is a broken command.

    `pip install ./sparam-lint` works in the development tree and fails for
    everyone else with "File './sparam-lint' does not exist". Every install
    line here has to resolve for someone who cloned only this repository.
    """
    readme = (_REPO / "README.md").read_text(encoding="utf-8")
    offenders = []
    for i, line in enumerate(readme.splitlines(), 1):
        if "pip install" not in line:
            continue
        for tok in line.split():
            tok = tok.strip("`\"',")
            if tok.startswith(("./", "../")) and not (_REPO / tok).exists():
                offenders.append("line %d: %s" % (i, tok))
    assert not offenders, (
        "install commands reference paths absent from a fresh clone: "
        + "; ".join(offenders)
    )


def test_pypi_install_lines_are_labelled_as_not_yet_available():
    """The package name 404s on PyPI today. Saying so is the whole fix."""
    readme = (_REPO / "README.md").read_text(encoding="utf-8")
    if "pip install maxwell-lint" in readme:
        assert "Not yet on PyPI" in readme, (
            "README shows `pip install maxwell-lint` without saying the name "
            "is not published yet"
        )


# -------------------------------------------- matrix mode + documentation

_EX = _REPO / "examples"


def _run_cli(argv):
    buf = io.StringIO()
    old, sys.stdout = sys.stdout, buf
    try:
        rc = cli_main(argv)
    finally:
        sys.stdout = old
    return rc, buf.getvalue()


def test_matrix_mode_agrees_with_both_ways_of_giving_the_reference():
    """--layout computes the isolated matrix; --isolated supplies it. Same verdict."""
    rc_a, out_a = _run_cli(["matrix", "--full", str(_EX / "born2.csv"),
                            "--layout", str(_EX / "geometry.csv"), "--json"])
    rc_b, out_b = _run_cli(["matrix", "--full", str(_EX / "born2.csv"),
                            "--isolated", str(_EX / "isolated.csv"), "--json"])
    a, b = json.loads(out_a), json.loads(out_b)
    assert rc_a == rc_b == 1
    assert a["n_violations"] == b["n_violations"]
    assert a["max_k"] == pytest.approx(b["max_k"], rel=1e-12)


def test_matrix_mode_refuses_without_a_reference():
    """k is a ratio. No denominator, no verdict -- and no guess."""
    with pytest.raises(SystemExit) as exc:
        _run_cli(["matrix", "--full", str(_EX / "born2.csv")])
    assert exc.value.code == 2


@pytest.mark.parametrize("name,expect_rc", [("closure.csv", 0), ("born2.csv", 1)])
def test_readme_matrix_transcript_is_real_output(name, expect_rc):
    readme = (_REPO / "README.md").read_text(encoding="utf-8")
    cmd = f"$ maxwell-lint matrix --full examples/{name} --layout examples/geometry.csv\n"
    documented = readme.split(cmd, 1)[1].split("\n$", 1)[0].split("```", 1)[0].strip()
    rc, out = _run_cli(["matrix", "--full", f"examples/{name}",
                        "--layout", "examples/geometry.csv", "--no-colour"])
    assert rc == expect_rc
    assert out.strip() == documented, f"transcript for {name} drifted from the code"


def test_example_matrices_regenerate_identically():
    """A fixture nobody can rebuild is a fixture nobody can trust."""
    import numpy as np

    from maxwell_lint.models import (
        born_second_order,
        isolated_pair_matrix,
        monopole_closure,
        random_layout,
    )
    lay = random_layout(8, seed=1, pitch_um=60.0, diameter_um=40.0)
    for name, arr in [("closure.csv", monopole_closure(lay)),
                      ("born2.csv", born_second_order(lay)),
                      ("isolated.csv", isolated_pair_matrix(lay))]:
        on_disk = np.loadtxt(_EX / name, delimiter=",")
        assert on_disk == pytest.approx(arr, rel=1e-10), f"{name} is stale"


def test_readme_cli_reference_covers_every_flag():
    import maxwell_lint.cli as cli
    src = pathlib.Path(cli.__file__).read_text(encoding="utf-8")
    flags = set(re.findall(r'p\.add_argument\("(--[a-z-]+)"', src))
    assert flags, "no flags found -- this guard has stopped looking"
    table = (_REPO / "README.md").read_text(encoding="utf-8").split(
        "## CLI reference", 1)[1].split("## Troubleshooting", 1)[0]
    for flag in flags:
        assert flag in table, f"{flag} exists but the CLI reference omits it"


def test_readme_api_table_names_only_real_objects():
    import maxwell_lint as M
    for name in ("Layout", "random_layout", "isolated_pair_matrix",
                 "monopole_closure", "born_second_order", "mean_field",
                 "screening_factor", "screening_depth", "pairwise_error",
                 "check_ceiling"):
        assert hasattr(M, name), f"README promises maxwell_lint.{name}"
    rep = M.check_ceiling(M.monopole_closure(M.random_layout(6, seed=0)),
                          M.isolated_pair_matrix(M.random_layout(6, seed=0)))
    for attr in ("n_pairs", "n_violations", "violation_fraction", "max_k",
                 "worst_pair", "median_depth", "median_pairwise_error",
                 "passed", "tol", "as_dict", "summary"):
        assert hasattr(rep, attr), f"README promises CeilingReport.{attr}"


# ------------------------------------------------ vectorized isolated pairs

def _isolated_pair_matrix_by_explicit_inverse(lay):
    """The pre-vectorization implementation, kept as the equivalence oracle.

    It builds each pair's 2x2 potential matrix and inverts it with
    ``np.linalg.inv``. Slow, but obviously correct -- which is exactly what an
    oracle is for.
    """
    from maxwell_lint.models import EPS0
    n = lay.n
    out = np.zeros((n, n))
    d = lay.distances()
    scale = 1.0 / (2.0 * np.pi * EPS0 * lay.eps_r)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = np.array([
                [-scale * np.log(lay.radius[i]), -scale * np.log(d[i, j])],
                [-scale * np.log(d[i, j]), -scale * np.log(lay.radius[j])],
            ])
            out[i, j] = abs(np.linalg.inv(p)[0, 1])
    return out


@pytest.mark.parametrize("n", [2, 3, 8, 33])
def test_isolated_pair_matrix_matches_the_explicit_two_by_two_inverse(n):
    """The closed form must equal the inverse it replaces, not merely track it."""
    lay = random_layout(n, seed=n)
    fast = isolated_pair_matrix(lay)
    slow = _isolated_pair_matrix_by_explicit_inverse(lay)
    assert fast.shape == slow.shape
    assert np.allclose(fast, slow, rtol=0, atol=1e-24), (
        f"worst absolute difference {np.max(np.abs(fast - slow)):.3e}"
    )


def test_isolated_pair_matrix_is_finite_and_has_a_zero_diagonal():
    """A conductor is not coupled to itself, and no inf/nan may escape.

    The vectorized form divides by a determinant that is -inf on the diagonal,
    so the diagonal has to be written rather than merely inherited.
    """
    lay = random_layout(12, seed=3)
    m = isolated_pair_matrix(lay)
    assert np.isfinite(m).all(), "non-finite value leaked out of the closed form"
    assert np.all(np.diag(m) == 0.0)
    assert np.all(m >= 0.0)


def test_isolated_pair_matrix_is_symmetric():
    """Pair (i,j) and (j,i) see the same two conductors, so must agree."""
    lay = random_layout(9, seed=5)
    m = isolated_pair_matrix(lay)
    assert np.allclose(m, m.T, rtol=0, atol=1e-24)


def test_unequal_radii_are_not_collapsed():
    """The closed form uses r_i and r_j separately; a bug could use one twice."""
    from maxwell_lint.models import Layout
    xy = np.array([[0.0, 0.0], [100e-6, 0.0]])
    lay = Layout(xy, np.array([20e-6, 5e-6]))
    assert isolated_pair_matrix(lay) == pytest.approx(
        _isolated_pair_matrix_by_explicit_inverse(lay), rel=0, abs=1e-24)
