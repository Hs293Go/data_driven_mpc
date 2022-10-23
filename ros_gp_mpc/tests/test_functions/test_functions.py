import numpy as np


def x_by_sin_x(x):
    return x * np.sin(x)


def rosenbrock(x) -> float:
    return np.sum(100 * (x[1::] - x[:-1:] ** 2) ** 2 + (1 - x[:-1:]) ** 2)
