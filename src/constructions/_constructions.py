import sympy as sp
import numpy as np
from jutility import plotting

def dot(x, y):
    return x.T * y

def l2_sq(x):
    return x.T * x

def l2(x):
    return sp.sqrt(x.T * x)

class Point:
    def __init__(self, coords):
        self.coords = coords

    def plot(self, **kwargs):
        return plotting.Scatter(*self.coords, **kwargs)

class Line:
    def __init__(self, a, b):
        self.a = a.coords
        self.b = b.coords
        self.bma = self.b - self.a
        self.bma_l2_sq = l2_sq(self.bma)

    def plot(self, **kwargs):
        a = np.array(self.a).flatten().astype(float)
        b = np.array(self.b).flatten().astype(float)
        return plotting.AxLine(a, b, **kwargs)

class Circle:
    def __init__(self, centre, r_sq):
        self.centre = centre.coords
        self.r_sq = r_sq
