from abc import ABC, abstractmethod
import numpy as np
import numpy.linalg as la
import typing as tp


class Metric(ABC):
    @staticmethod
    def is_vector(x: np.ndarray):
        return len(x.shape) == 1

    @abstractmethod
    def __call__(self, x: tp.Any, y: tp.Any):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def additional(self):
        pass


class CosineMetric(Metric):

    def additional(self):
        return None

    def __str__(self):
        return 'cosine'

    def __call__(self, x: np.ndarray, y: np.ndarray):
        assert Metric.is_vector(x) and Metric.is_vector(y)
        return 1 - (x @ y) / (la.norm(x, ord=2) * la.norm(y, ord=2))


class MinkowskiMetric(Metric):
    def __init__(self, p):
        self.p = p

    def additional(self):
        return self.p

    def __str__(self):
        return 'minkowski'

    def __call__(self, x: np.ndarray, y: np.ndarray):
        assert Metric.is_vector(x) and Metric.is_vector(y)
        return la.norm(x - y, ord=self.p)


class JacquardMetric(Metric):

    def __str__(self):
        return 'jaccard'

    def additional(self):
        return None

    def __call__(self, x: set, y: set):
        assert type(x) is set and type(y) is set
        return 1 - len(x & y) / len(x | y)
