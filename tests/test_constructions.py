import os
import sympy as sp
import numpy as np
from jutility import plotting, util
import constructions as cn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "Outputs")

util.numpy_set_print_options()

def test_plot():
    rng = util.Seeder().get_rng("test_plot")
    a, b, c, d = [rng.integers(-10, 10, 2) for _ in range(4)]

    a = cn.Point(sp.Matrix(a))
    b = cn.Point(sp.Matrix(b))
    c = cn.Point(sp.Matrix(c))
    d = cn.Point(sp.Matrix(d))
    ab_line = cn.Line(a, b)
    cd_line = cn.Line(c, d)

    plotting.plot(
        *[p.plot(c="r", zorder=20) for p in [a, b, c, d]],
        *[line.plot(c="b") for line in [ab_line, cd_line]],
        figsize=[8, 6],
        axis_equal=True,
        plot_name="test_plot",
        dir_name=RESULTS_DIR,
    )
