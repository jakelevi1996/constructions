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
        m_point = Point(self.a + alpha * self.bma)
        return m_point

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
        self.centre = centre_point.coords
        self.r_sq = sp.simplify(r_sq)
        if self.r_sq <= 0:
            raise ValueError("`r_sq` must be > 0, received %s" % self.r_sq)

    def contains_point(self, point):
        centre_distance = Point(self.centre).l2_sq_distance(point)
        return sp.simplify(centre_distance - self.r_sq) == 0

    def plot(self, **kwargs):
        return plotting.Circle(self.centre, sp.sqrt(self.r_sq), **kwargs)

    def __repr__(self):
        r = sp.sqrt(self.r_sq)
        return "Circle(centre=%s, r=%s)" % (tuple(self.centre), r)
