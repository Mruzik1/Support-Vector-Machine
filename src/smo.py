import numpy as np
from numpy import ndarray
from random import choice


class SMO:
    def __init__(self, X: ndarray, y: ndarray, kernel: str, c: float = 1, tol: float = 10e-4, gamma: float = 0.1):
        self._X = X
        self._y = y
        self._C = c
        self._tol = tol
        self._set_kernel_type(kernel, gamma)

        np.random.seed(42)
        self._alphas = np.random.uniform(0, c, size=len(y))
        self._b = 0
    
    # for a more organised initialization
    def _set_kernel_type(self, kernel: str, gamma: float):
        if kernel.lower() == 'linear':
            self._kernel = self._linear_kernel
        elif kernel.lower() == 'rbf':
            self._kernel = self._rbf_kernel
            self._gamma = gamma
        else:
            raise AttributeError(f'A kernel type "{kernel}" does not exist!')
    
    # a linear kernel function
    def _linear_kernel(self, x1: ndarray, x2: ndarray) -> float:
        return np.inner(x1, x2)

    # a radial basis kernel function
    def _rbf_kernel(self, x1: ndarray, x2: ndarray) -> float:
        x1 = x1[np.newaxis, :] if np.ndim(x1) == 1 else x1
        x2 = x2[np.newaxis, :] if np.ndim(x2) == 1 else x2
        euclidean_dist = np.linalg.norm(x1[:, :, np.newaxis] - x2.T[np.newaxis, :, :], axis=1)**2

        return np.exp(-self._gamma * np.squeeze(euclidean_dist))

    # getting one prediction
    def _get_prediction(self, x: ndarray) -> float:
        return np.sum(self._alphas*self._y*self._kernel(self._X, x)) + self._b
    
    # getting the bounds (considering our constraints)
    def _get_bounds(self, i: int, j: int) -> tuple:
        if self._y[i] != self._y[j]:
            L = max(0, self._alphas[j]-self._alphas[i])
            H = min(self._C, self._C+self._alphas[j]-self._alphas[i])
        else:
            L = max(0, self._alphas[i]+self._alphas[j]-self._C)
            H = min(self._C, self._alphas[i]+self._alphas[j])
        return L, H

    # calculating an error
    def _get_error(self, i: int) -> float:
        return self._get_prediction(self._X[i]) - self._y[i]

    # calculating eta
    def _get_eta(self, x1: ndarray, x2: ndarray) -> float:
        return 2*self._kernel(x1, x2) - self._kernel(x1, x1) - self._kernel(x2, x2)

    # clipping a new alpha wrt the bounds
    def _clip_alpha(self, L: float, H: float, alpha: float) -> float:
        if alpha >= H:
            return H
        elif alpha <= L:
            return L
        return alpha

    # calculating a threshold
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

    # one step of SMO
    def _step(self, i: int, j: int, e1: float) -> bool:
        L, H = self._get_bounds(i, j)
        e2 = self._get_error(j)
        eta = self._get_eta(self._X[i], self._X[j])

        if eta >= 0 or L >= H:
            return False
        
        new_a2 = self._clip_alpha(L, H, self._alphas[j]-(self._y[j]*(e1-e2))/eta)
        new_a1 = self._alphas[i]+self._y[i]*self._y[j]*(self._alphas[j]-new_a2)

        self._b = self._get_threshold(i, j, new_a1, new_a2, e1, e2)
        self._alphas[i] = new_a1
        self._alphas[j] = new_a2
        return True

    # getting predictions for every datapoint in the training dataset
    def predict(self, features: ndarray, labels: ndarray) -> tuple:
        pred = np.array([self._get_prediction(x) for x in features])
        acc = np.count_nonzero(np.sign(pred) == labels) / labels.shape[0]
        return pred, acc

    # the fitting method
    def fit(self) -> tuple:
        for i, a in enumerate(self._alphas):
            e = self._get_error(i)
            if (self._y[i]*e < -self._tol and a < self._C) or (self._y[i]*e > self._tol and a > 0):
                j = choice(list(range(i)) + list(range(i+1, len(self._y))))
                self._step(i, j, e)
        return self.predict(self._X, self._y)

    # getting support vectors
    def get_support_vectors(self, zero: float = 10e-5) -> ndarray:
        return self._X[self._alphas > zero]