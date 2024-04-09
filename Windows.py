from abc import ABC, abstractmethod
import numpy as np
from Metrics import Metric


class Window(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class FixedWindow(Window):

    def __init__(self, h: np.float64):
        self.h = h

    def __call__(self, *args, **kwargs):
        return self.h


class NonFixedWindow(Window):

    def __call__(self, metric: Metric, x_k: np.float64, u: np.float64):
        return metric(x_k, u)
