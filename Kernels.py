from abc import ABC, abstractmethod
import numpy as np
import numpy.linalg as la
import typing as tp
import scipy.stats as sps


class Kernel(ABC):

    @staticmethod
    def _is_norm(x):
        return 1 if abs(x) < 1 else 0

    @abstractmethod
    def __call__(self, x):
        pass


class RectangularKernel(Kernel):
    def __call__(self, x: np.float64):
        return 0.5 * self._is_norm(x)


class TriangularKernel(Kernel):
    def __call__(self, x):
        return (1 - abs(x)) * self._is_norm(x)


class PolynomialKernel(Kernel):

    def __init__(self, a: np.float64, b: np.float64):
        self.a = a
        self.b = b

    def __call__(self, x):
        return np.power(1 - np.power(abs(x), self.a), self.b)


class GaussianKernel(Kernel):

    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale

    def __call__(self, x):
        return sps.norm.pdf(x, loc=self.loc, scale=self.scale)
