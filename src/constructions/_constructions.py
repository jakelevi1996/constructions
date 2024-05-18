import sympy as sp

def dot(x, y):
    return x.T * y

def l2_sq(x):
    return x.T * x

def l2(x):
    return sp.sqrt(x.T * x)

class Point:
    def __init__(self, coords):
        self.coords = coords

class Line:
    def __init__(self, a, b):
        self.a = a.coords
        self.b = b.coords
        self.bma = self.b - self.a
        self.bma_l2_sq = l2_sq(self.bma)

class Circle:
    def __init__(self, centre, r_sq):
        self.centre = centre.coords
        self.r_sq = r_sq
