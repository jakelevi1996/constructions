import sympy as sp
import numpy as np
from jutility import plotting

ROTATE_90 = sp.Matrix([[0, -1], [1, 0]])

def dot(x, y):
    [z] = x.T * y
    return z

def l2_sq(x):
    return dot(x, x)

def l2(x):
    return sp.sqrt(dot(x, x))

class Point:
    def __init__(self, coords_matrix):
        self.coords = coords_matrix

    def l2_sq_distance(self, other_point):
        return l2_sq(other_point.coords - self.coords)

    def plot(self, **kwargs):
        return plotting.Scatter(*self.coords, **kwargs)

    def __repr__(self):
        return "Point(%s, %s)" % tuple(self.coords)

class Line:
    def __init__(self, a_point, b_point):
        self.a = a_point.coords
        self.b = b_point.coords
        self.bma = self.b - self.a
        self.bma_l2_sq = sp.simplify(l2_sq(self.bma))
        if self.bma_l2_sq == 0:
            raise ValueError(
                "Input points %s and %s must not be equal"
                % (a_point, b_point)
            )

    def project_point(self, point):
        alpha = dot(point.coords - self.a, self.bma) / self.bma_l2_sq
        return Point(self.a + alpha * self.bma)

    def contains_point(self, point):
        m = self.project_point(point)
        return sp.simplify(m.l2_sq_distance(point)) == 0

    def is_direction_orthogonal(self, d):
        return sp.simplify(dot(d, self.bma)) == 0

    def get_direction_orthogonal_component(self, d):
        alpha = dot(d, self.bma) / self.bma_l2_sq
        return d - alpha * self.bma

    def is_line_parallel(self, line):
        x = self.get_direction_orthogonal_component(line.bma)
        return sp.simplify(l2_sq(x)) == 0

    def get_intersection_line(self, line):
        if self.is_line_parallel(line):
            return []

        alpha, _ = sp.Matrix([[self.bma, line.bma]]).solve(line.a - self.a)
        return [Point(self.a + alpha * self.bma)]

    def get_intersection_circle(self, circle):
        m = self.project_point(circle.centre)
        alpha = (
            (circle.r_sq - m.l2_sq_distance(circle.centre)) / self.bma_l2_sq
        )
        if alpha < 0:
            return []
        if alpha == 0:
            return [m]

        half_chord = sp.sqrt(alpha) * self.bma
        return [Point(m.coords + half_chord), Point(m.coords - half_chord)]

    def get_points(self):
        return Point(self.a), Point(self.b)

    def plot(self, **kwargs):
        a = np.array(self.a).flatten().astype(float)
        b = np.array(self.b).flatten().astype(float)
        return plotting.AxLine(a, b, **kwargs)

    def __repr__(self):
        return "Line(through=[%s and %s])" % (tuple(self.a), tuple(self.b))

class Circle:
    def __init__(self, centre_point, r_sq):
        self.centre = centre_point
        self.r_sq = sp.simplify(r_sq)
        if self.r_sq <= 0:
            raise ValueError("`r_sq` must be > 0, received %s" % self.r_sq)

    def contains_point(self, point):
        centre_distance = self.centre.l2_sq_distance(point)
        return sp.simplify(centre_distance - self.r_sq) == 0

    def get_intersection_circle(self, circle):
        centre_disp = circle.centre.coords - self.centre.coords
        cd_l2_sq = sp.simplify(l2_sq(centre_disp))
        if cd_l2_sq == 0:
            return []

        alpha = (self.r_sq + cd_l2_sq - circle.r_sq) / (2 * cd_l2_sq)
        m = self.centre.coords + alpha * centre_disp
        beta = self.r_sq / cd_l2_sq - alpha * alpha
        if beta < 0:
            return []
        if beta == 0:
            return [Point(m)]

        half_chord = sp.sqrt(beta) * (ROTATE_90 * centre_disp)
        return [Point(m + half_chord), Point(m - half_chord)]

    def plot(self, **kwargs):
        r = sp.sqrt(self.r_sq)
        return plotting.Circle(self.centre.coords, r, **kwargs)

    def __repr__(self):
        r = sp.sqrt(self.r_sq)
        return "Circle(centre=%s, r=%s)" % (tuple(self.centre.coords), r)
