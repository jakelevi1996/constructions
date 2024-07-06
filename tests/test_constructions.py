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
    num_lo = math.ceil( denom * lo) + 1
    num_hi = math.floor(denom * hi) - 1
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

def random_line(*args, **kwargs):
    a = random_point(*args, **kwargs)
    b = random_point(*args, **kwargs)
    return cn.Line(a, b)

def random_rotation(rng, cos_lo=-1, cos_hi=1, denom_lo=50, denom_hi=100):
    c = random_rational(rng, cos_lo, cos_hi, denom_lo, denom_hi)
    s = sp.sqrt(1 - c*c)
    r = sp.Matrix([[c, -s], [s, c]])
    return r

def sum_sqrt(a, b):
    return sp.sqrt(a) + sp.sqrt(b)

def sqrt_sq_sum_sqrt(a, b):
    return sp.sqrt(a + b + 2*sp.sqrt(a*b))

def test_valid_line():
    rng = util.Seeder().get_rng("test_valid_line")
    printer = util.Printer("test_valid_line", RESULTS_DIR)
    a_coords = [
        sum_sqrt(2, 3),
        sqrt_sq_sum_sqrt(5, 7),
    ]
    b_coords = [
        sqrt_sq_sum_sqrt(2, 3),
        sum_sqrt(5, 7),
    ]
    a = cn.Point(sp.Matrix(a_coords))
    b = cn.Point(sp.Matrix(b_coords))

    with pytest.raises(ValueError):
        line = cn.Line(a, b)

    r = random_rotation(rng, cos_lo=0.3, cos_hi=0.6)
    c = cn.Point(r * b.coords)
    line = cn.Line(a, c)

    printer(a, b, r, c, line, sep="\n")

def test_valid_circle():
    rng = util.Seeder().get_rng("test_valid_circle")
    printer = util.Printer("test_valid_circle", RESULTS_DIR)

    a = random_point(rng)
    r_sq = sum_sqrt(2, 3) - sqrt_sq_sum_sqrt(2, 3)

    assert r_sq != 0
    assert sp.simplify(r_sq) == 0

    with pytest.raises(ValueError):
        circle = cn.Circle(a, r_sq)

    circle = cn.Circle(a, random_rational(rng, 1.1, 1.9))

    printer(a, r_sq, circle, sep="\n")

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

    line = random_line(rng)
    alpha = random_rational(rng, 1.1, 1.9)
    c = cn.Point(line.a + alpha * line.bma)
    assert line.contains_point(c)

    r = random_rotation(rng, cos_lo=0.1, cos_hi=0.3)
    d = cn.Point(line.a + alpha * (r * line.bma))
    assert not line.contains_point(d)

    a, b = line.get_points()
    printer(a, b, c, d, alpha, r, sep="\n")
    plotting.plot(
        *[p.plot(c="r", s=100, zorder=20) for p in [a, b, c, d]],
        line.plot(c="b"),
        cn.Circle(a, a.l2_sq_distance(c)).plot(c="m"),
        axis_equal=True,
        plot_name=test_name,
        dir_name=RESULTS_DIR,
    )

@pytest.mark.parametrize("seed", range(3))
def test_line_project_point(seed):
    test_name = "test_line_project_point_%i" % seed
    rng = util.Seeder().get_rng(test_name)
    printer = util.Printer(test_name, RESULTS_DIR)

    line = random_line(rng)
    alpha = random_rational(rng, 1.1, 1.9)
    r = random_rotation(rng, cos_lo=0.3, cos_hi=0.6)
    c = cn.Point(line.a + alpha * (r * line.bma))
    m = line.project_point(c)

    assert line.contains_point(m)

    eps = sp.Rational(1, 100) * line.bma
    assert c.l2_sq_distance(m) < c.l2_sq_distance(cn.Point(m.coords + eps))
    assert c.l2_sq_distance(m) < c.l2_sq_distance(cn.Point(m.coords - eps))

    assert line.is_direction_orthogonal(m.coords - c.coords)

    a, b = line.get_points()
    printer(a, b, line, alpha, r, c, m, eps, sep="\n")
    plotting.plot(
        *[p.plot(c="r", s=100, zorder=20) for p in [a, b, c, m]],
        line.plot(c="b"),
        cn.Circle(c, c.l2_sq_distance(m)).plot(c="m"),
        axis_equal=True,
        plot_name=test_name,
        dir_name=RESULTS_DIR,
    )

@pytest.mark.parametrize("seed", range(3))
def test_line_is_direction_orthogonal(seed):
    test_name = "test_line_is_direction_orthogonal_%i" % seed
    rng = util.Seeder().get_rng(test_name)
    printer = util.Printer(test_name, RESULTS_DIR)

    line = random_line(rng)
    alpha = random_rational(rng, 1.1, 1.9)
    d1 = alpha * cn.ROTATE_90 * line.bma

    assert line.is_direction_orthogonal(d1)

    r = random_rotation(rng, cos_lo=0.3, cos_hi=0.6)
    d2 = r * d1

    assert not line.is_direction_orthogonal(d2)

    printer(line, alpha, r, d1, d2, sep="\n")

@pytest.mark.parametrize("seed", range(3))
def test_line_get_intersection_line(seed):
    test_name = "test_line_get_intersection_line_%i" % seed
    rng = util.Seeder().get_rng(test_name)
    printer = util.Printer(test_name, RESULTS_DIR)

    line1 = random_line(rng)
    alpha = random_rational(rng, 1.1, 1.9)
    beta  = random_rational(rng, 1.1, 1.9)
    r1 = random_rotation(rng, cos_lo=0.3, cos_hi=0.6)
    r2 = random_rotation(rng, cos_lo=0.3, cos_hi=0.6)
    c = cn.Point(line1.a + alpha * r1 * line1.bma)
    d = cn.Point(c.coords + beta * line1.bma)
    e = cn.Point(c.coords + beta * r2 * line1.bma)
    line2 = cn.Line(c, d)
    line3 = cn.Line(c, e)

    assert line1.is_line_parallel(line2)
    assert not line1.is_line_parallel(line3)

    x12_list = line1.get_intersection_line(line2)
    x13_list = line1.get_intersection_line(line3)

    assert len(x12_list) == 0
    assert len(x13_list) == 1

    [x] = x13_list

    assert line1.contains_point(x)
    assert line3.contains_point(x)
    assert not line2.contains_point(x)

    a, b = line1.get_points()
    plotting.plot(
        *[line.plot(c="b") for line in [line1, line2, line3]],
        *[p.plot(c="g", s=100, zorder=20) for p in [a, b, c, d, e]],
        x.plot(c="r", s=100, zorder=20),
        axis_equal=True,
        plot_name=test_name,
        dir_name=RESULTS_DIR,
    )

    printer(a, b, c, d, e, line1, line2, line3, x, sep="\n")

@pytest.mark.parametrize("seed", range(3))
def test_line_get_intersection_circle(seed):
    test_name = "test_line_get_intersection_circle_%i" % seed
    rng = util.Seeder().get_rng(test_name)
    printer = util.Printer(test_name, RESULTS_DIR)

    line = random_line(rng)
    c = random_point(rng)
    m = line.project_point(c)
    d = m.l2_sq_distance(c)
    alpha_list = [
        random_rational(rng, 0.2, 0.6),
        1,
        random_rational(rng, 1.2, 1.6),
    ]
    circles = [
        cn.Circle(c, alpha * d)
        for alpha in alpha_list
    ]
    x = [
        line.get_intersection_circle(circle)
        for circle in circles
    ]

    num_x = [0, 1, 2]
    assert len(num_x) == len(x)
    assert all(num_xi == len(xi) for num_xi, xi in zip(num_x, x))
    assert all(line.contains_point(xii) for xi in x for xii in xi)

    assert circles[1].contains_point(x[1][0])
    assert circles[2].contains_point(x[2][0])
    assert circles[2].contains_point(x[2][1])
    assert all(
        c.contains_point(xii)
        for xi, c in zip(x, circles)
        for xii in xi
    )

    plotting.plot(
        line.plot(c="b"),
        c.plot(c="g", s=100, zorder=20),
        *[c.plot(c="m") for c in circles],
        *[xii.plot(c="r", s=100, zorder=20) for xi in x for xii in xi],
        axis_equal=True,
        grid=False,
        plot_name=test_name,
        dir_name=RESULTS_DIR,
    )
    printer(line, *circles, *x, sep="\n\n")

@pytest.mark.parametrize("seed", range(3))
def test_circle_get_intersection_circle(seed):
    test_name = "test_circle_get_intersection_circle_%i" % seed
    rng = util.Seeder().get_rng(test_name)
    printer = util.Printer(test_name, RESULTS_DIR)

    p1 = random_point(rng)
    p2 = random_point(rng)
    d = sp.sqrt(p1.l2_sq_distance(p2))
    r1 = random_rational(rng, 0.4, 0.6) * d
    c1 = cn.Circle(p1, r1*r1)
    r2_list = [
        random_rational(rng, 0.4, 0.6) * (d - r1),
        (d - r1),
        random_rational(rng, (d - r1), d),
        d,
        random_rational(rng, d, 0.9*(d + r1)),
        random_rational(rng, 0.9, 1) * (d + r1),
        (d + r1),
        random_rational(rng, 1.2, 1.4) * (d + r1),
    ]
    c2_list = [
        cn.Circle(p2, r2*r2)
        for r2 in r2_list
    ]
    x = [
        c1.get_intersection_circle(c2) for c2 in c2_list
    ]
    p3 = cn.Point(p2.coords + r2_list[-1] * (p1.coords - p2.coords) / d)

    num_x = [0, 1, 2, 2, 2, 2, 1, 0,]
    assert len(num_x) == len(x)
    assert all(num_xi == len(xi) for num_xi, xi in zip(num_x, x))
    assert all(c1.contains_point(xii) for xi in x for xii in xi)
    assert all(
        c2.contains_point(xii)
        for xi, c2 in zip(x, c2_list)
        for xii in xi
    )

    plotting.plot(
        c1.plot(c="b"),
        *[c2.plot(c="m") for c2 in c2_list],
        *[xii.plot(c="r", s=100, zorder=20) for xi in x for xii in xi],
        p1.plot(c="g"),
        p2.plot(c="g"),
        p3.plot(c="c"),
        axis_equal=True,
        grid=False,
        plot_name=test_name,
        dir_name=RESULTS_DIR,
    )
    printer(c1, *c2_list, *x, sep="\n\n")

@pytest.mark.parametrize("seed", range(3))
def test_circle_contains_point(seed):
    test_name = "test_circle_contains_point_%i" % seed
    rng = util.Seeder().get_rng(test_name)
    printer = util.Printer(test_name, RESULTS_DIR)

    a, b = [random_point(rng) for _ in range(2)]
    circle = cn.Circle(a, a.l2_sq_distance(b))
    r = random_rotation(rng, cos_lo=-0.5, cos_hi=0.5)
    c = cn.Point(a.coords + r * (b.coords - a.coords))
    assert circle.contains_point(c)

    alpha = random_rational(rng, 1.1, 1.9)
    d = cn.Point(a.coords + alpha * r * (b.coords - a.coords))
    assert not circle.contains_point(d)

    printer(a, b, circle, r, c, d, sep="\n")
    plotting.plot(
        *[p.plot(c="r", s=100, zorder=20) for p in [a, b, c, d]],
        circle.plot(c="m"),
        axis_equal=True,
        plot_name=test_name,
        dir_name=RESULTS_DIR,
    )

@pytest.mark.parametrize("seed", range(3))
def test_point_set(seed):
    test_name = "test_point_set_%i" % seed
    rng = util.Seeder().get_rng(test_name)
    printer = util.Printer(test_name, RESULTS_DIR)

    a_coords = [
        sum_sqrt(2, 3),
        sqrt_sq_sum_sqrt(5, 7),
    ]
    b_coords = [
        sqrt_sq_sum_sqrt(2, 3),
        sum_sqrt(5, 7),
    ]
    a = cn.Point(sp.Matrix(a_coords))
    b = cn.Point(sp.Matrix(b_coords))

    assert a == b
    assert a != random_point(rng)

    s = set([a, b])

    assert len(s) == 1

    printer(a, b, s, len(s), sep="\n\n")

    p1 = random_point(rng)
    p2 = random_point(rng)
    d = sp.sqrt(p1.l2_sq_distance(p2))
    r1 = random_rational(rng, 0.4, 0.6) * d
    r2 = random_rational(rng, (d - r1), (d + r1))
    c1 = cn.Circle(p1, r1*r1)
    c2 = cn.Circle(p2, r2*r2)
    x = c1.get_intersection_circle(c2) + c2.get_intersection_circle(c1)
    s = set(x)

    assert len(x) == 4
    assert len(s) == 2

    plotting.plot(
        c1.plot(c="b"),
        c2.plot(c="b"),
        *[xi.plot(c="r", s=100, zorder=20) for xi in x],
        p1.plot(c="g"),
        p2.plot(c="g"),
        axis_equal=True,
        grid=False,
        plot_name=test_name,
        dir_name=RESULTS_DIR,
    )

    printer(c1, c2, x, s, len(s), sep="\n\n")

@pytest.mark.parametrize("seed", range(3))
def test_line_set(seed):
    test_name = "test_line_set_%i" % seed
    rng = util.Seeder().get_rng(test_name)
    printer = util.Printer(test_name, RESULTS_DIR)

    s1 = random_line(rng)
    s2 = random_line(rng)
    p1, p2 = s2.get_points()
    p1_proj = s1.project_point(p1)
    p2_proj = s1.project_point(p2)
    s3 = cn.Line(p1_proj, p2_proj)

    assert s1 != s2
    assert s1 == s3
    assert len(set([s1]))           == 1
    assert len(set([s1, s2]))       == 2
    assert len(set([s1, s3]))       == 1
    assert len(set([s1, s2, s3]))   == 2

    plotting.plot(
        s1.plot(),
        s2.plot(),
        s3.plot(lw=20, z=0, alpha=0.5, c="g"),
        cn.Line(p1, p1_proj).plot(c="g"),
        cn.Line(p2, p2_proj).plot(c="g"),
        *[p.plot(c="r", z=20, s=80) for p in [p1, p2, p1_proj, p2_proj]],
        *[p.plot(c="b", z=20, s=80) for p in s1.get_points()],
        axis_equal=True,
        plot_name=test_name,
        dir_name=RESULTS_DIR,
    )

    printer(s1, s2, s3, p1, p2, p1_proj, p2_proj, sep="\n\n")
