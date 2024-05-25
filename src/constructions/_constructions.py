import sympy as sp
import numpy as np
from jutility import plotting

def dot(x, y):
    return x.T * y

def l2_sq(x):
    [y] = x.T * x
    return y

def l2(x):
    return sp.sqrt(l2_sq(x))

class Point:
    def __init__(self, coords_matrix):
        self.coords = coords_matrix

    def l2_sq_distance(self, other_point):
        return l2_sq(other_point.coords - self.coords)

    def plot(self, **kwargs):
        return plotting.Scatter(*self.coords, **kwargs)

class Line:
    def __init__(self, a_point, b_point):
        self.a = a_point.coords
        self.b = b_point.coords
        self.bma = self.b - self.a
        self.bma_l2_sq = l2_sq(self.bma)

    def plot(self, **kwargs):
        a = np.array(self.a).flatten().astype(float)
        b = np.array(self.b).flatten().astype(float)
        return plotting.AxLine(a, b, **kwargs)

class Circle:
    def __init__(self, centre_point, r_sq):
        self.centre = centre_point.coords
        self.r_sq = r_sq

    def plot(self, **kwargs):
        return plotting.Circle(self.centre, sp.sqrt(self.r_sq), **kwargs)
