import numpy as np
from numpy import ndarray


class SMO:
    def __init__(self, X: ndarray, y: ndarray, kernel: str, c: float):
        self._X = X
        self._y = y
        self._C = c
        self._set_kernel_type(kernel)

        np.random.seed(42)
        self._alphas = np.random.uniform(0, c, size=len(y))
        self._b = 0
        
    def _set_kernel_type(self, kernel: str):
        if kernel == 'linear':
            self._kernel = self._linear_kernel
        else:
            raise AttributeError(f'An kernel type "{kernel}" does not exist!')
    
    def _linear_kernel(self, x1: ndarray, x2: ndarray) -> float:
        return np.inner(x1, x2)

    def _get_inner_product(self, x: ndarray) -> float:
        return np.sum(self._alphas*self._y*self._kernel(self._X, x)) + self._b
    
    def _get_bounds(self, i: int, j: int) -> tuple:
        if self._y[i] != self._y[j]:
            L = max(0, self._alphas[j]-self._alphas[i])
            H = min(self._C, self._C+self._alphas[j]-self._alphas[i])
        else:
            L = max(0, self._alphas[i]+self._alphas[j]-self._C)
            H = min(self._C, self._alphas[i]+self._alphas[j])
        return L, H

    def _get_error(self, i: int) -> float:
        return self._get_inner_product(self._X[i]) - self._y[i]

    def _get_eta(self, x1: ndarray, x2: ndarray) -> float:
        return 2*self._kernel(x1, x2) - self._kernel(x1, x1) - self._kernel(x2, x2)

    def _clip_alpha(self, L: float, H: float, alpha: float) -> float:
        if alpha >= H:
            return H
        elif alpha <= L:
            return L
        return alpha

    def _get_threshold(self, i: int, j: int, new_a1: float, new_a2: float, e1: float, e2: float) -> float:
        k_ii = self._kernel(self._X[i], self._X[i])
        k_jj = self._kernel(self._X[j], self._X[j])
        k_ij = self._kernel(self._X[i], self._X[j])

        b1 = self._b - e1 - self._y[i]*(new_a1-self._alphas[i])*k_ii - self._y[j]*(new_a2-self._alphas[j])*k_ij
        b2 = self._b - e2 - self._y[i]*(new_a1-self._alphas[i])*k_ij - self._y[j]*(new_a2-self._alphas[j])*k_jj

        if 0 < self._alphas[i] < self._C:
            return b1
        elif 0 < self._alphas[j] < self._C:
            return b2
        return (b1+b2)/2

    def _step(self, i: int, j: int):
        L, H = self._get_bounds(i, j)
        e1 = self._get_error(i)
        e2 = self._get_error(j)
        eta = self._get_eta(self._X[i], self._X[j])

        