import os
import math
import sympy as sp
import numpy as np
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
