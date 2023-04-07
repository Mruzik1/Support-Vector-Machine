import numpy as np
from enum import Enum
from numpy import ndarray


class KernelType(Enum):
    LINEAR = 0


class SMO:
    def __init__(self, X: ndarray, y: ndarray, kernel: KernelType, c: float):
        self._X = X
        self._y = y
        self._C = c
        
        if kernel == KernelType.LINEAR:
            self._kernel = self.linear_kernel
    
    def linear_kernel(self, x1: ndarray, x2: ndarray) -> ndarray:
        return np.inner(x1, x2)

    