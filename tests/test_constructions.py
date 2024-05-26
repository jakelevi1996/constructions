import os
import math
import sympy as sp
import numpy as np
import pytest
from jutility import plotting, util
import constructions as cn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "Outputs")

util.numpy_set_print_options()

def random_rational(rng, lo=-1, hi=1, denom_lo=50, denom_hi=100):
    denom = rng.integers(denom_lo, denom_hi)
    num_lo = math.ceil( denom * lo)
    num_hi = math.floor(denom * hi)
    num = rng.integers(num_lo, num_hi)
    rational = sp.Rational(num, denom)
    assert (lo <= rational) and (rational <= hi)
    return rational

def random_point(*args, **kwargs):
    x = random_rational(*args, **kwargs)
    y = random_rational(*args, **kwargs)
    m = sp.Matrix([x, y])
    p = cn.Point(m)
    return p

def random_rotation(rng, cos_lo=-1, cos_hi=1, denom_lo=50, denom_hi=100):
    c = random_rational(rng, cos_lo, cos_hi, denom_lo, denom_hi)
    s = sp.sqrt(1 - c*c)
    r = sp.Matrix([[c, -s], [s, c]])
    return r

def test_plot():
    rng = util.Seeder().get_rng("test_plot")
    a, b, c, d = [random_point(rng) for _ in range(4)]

    plotting.plot(
        *[p.plot(c="r", s=100, zorder=20) for p in [a, b, c, d]],
        *[s.plot(c="b") for s in [cn.Line(a, b), cn.Line(c, d)]],
        *[
            cn.Circle(a, a.l2_sq_distance(x)).plot(c="m")
            for x in [b, c, d]
        ],
        figsize=[8, 6],
        ylim=[-2, 2],
        xlim=[-2, 2],
        axis_equal=True,
        plot_name="test_plot",
        dir_name=RESULTS_DIR,
    )

def test_repr():
    rng = util.Seeder().get_rng("test_repr")
    printer = util.Printer("test_repr", RESULTS_DIR)

    a, b, c, d = [random_point(rng) for _ in range(4)]
    lines = [cn.Line(a, b), cn.Line(c, d)]
    circles = [
        cn.Circle(a, a.l2_sq_distance(x))
        for x in [b, c, d]
    ]
    printer(a, b, c, d,   sep="\n", end="\n\n")
    printer(*lines,       sep="\n", end="\n\n")
    printer(*circles,     sep="\n", end="\n\n")

@pytest.mark.parametrize("seed", range(3))
def test_line_contains_point(seed):
    test_name = "test_line_contains_point_%i" % seed
    rng = util.Seeder().get_rng(test_name)
    printer = util.Printer(test_name, RESULTS_DIR)

    a, b = [random_point(rng) for _ in range(2)]
    line = cn.Line(a, b)
    alpha = random_rational(rng, 1.1, 1.9)
    c = cn.Point(a.coords + alpha * line.bma)
    assert line.contains_point(c)

    r = random_rotation(rng, cos_lo=0.1, cos_hi=0.3)
    d = cn.Point(a.coords + alpha * (r * line.bma))
    assert not line.contains_point(d)

    printer(a, b, c, d, alpha, r, sep="\n")
    plotting.plot(
        *[p.plot(c="r", s=100, zorder=20) for p in [a, b, c, d]],
        line.plot(c="b"),
        cn.Circle(a, a.l2_sq_distance(c)).plot(c="m"),
        axis_equal=True,
        plot_name=test_name,
        dir_name=RESULTS_DIR,
    )
