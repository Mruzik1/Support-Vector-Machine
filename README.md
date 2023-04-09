# Abstract :^)
Here I implemented SVM from scratch (only numpy was used) and tried to train it with different data and hyperparameters. An optimization algorithm I use here is SMO (i.e. Sequential Minimal Optimization), which is commonly used with Support Vector Machine. Although my implementation is a simplified version of the algorithm.

Advantages of SMO:
  - **Scalability:** SMO is designed to work well with large datasets.
  - **Convergence:** SMO has a proven convergence guarantee, which means that it will converge to the optimal solution given enough iterations.
  - **Flexibility:** SMO can handle a wide range of kernel functions, including linear, polynomial, and RBF kernels.
  - **Memory efficiency:** SMO only needs to store a small number of variables in memory, which makes it memory-efficient.
  
<a href="https://chubakbidpaa.com/assets/pdf/smo.pdf">I used this paper to implement the algorithm.</a> Here you can learn how it works from a mathematical point of view. Or, if you by some chance are interested in a full version of SMO, <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf">here</a> you can read a paper about it. 

# A Brief Overview On How My Script Works
In this section I will quickly explain how to work with my SVM implementation.

## SMO Class' Init Method
```python
def __init__(X: ndarray, y: ndarray, kernel: str, c: float = 1, tol: float = 10e-4, sigma: float = 1)
```

- **X** - a 2D numpy array; the first dimention contains data points, the second one contains the features.
- **y** - a 1D numpy array; consists of the labels for every data point in **X**; those labels should be either 1 or -1 (only binary classification).
- **kernel** - a string; "linear" or "rbf" (i.e names of the kernel functions).
- **C** - regularization parameter (just read the paper).
- **tol** - numerical tolerance (just read the paper).
- **sigma** - a hyperparameter for the RBF kernel, isn't needed while using the linear kernel function.

## Making Predictions
```python
def predict(features: ndarray, labels: ndarray) -> tuple[ndarray, float]
```

- **features** - a 2D numpy array; the first dimention contains data points, the second one contains the features.
- **labels** - a 1D numpy array; consists of the labels for every data point in **features**; those labels should be either 1 or -1 (only binary classification).
- **RETURNS** - a tuple with the predictions and an accuracy respectively.

## Training a Model
```python
def fit() -> tuple[ndarray, float]
```
- **BEHAVIOR** - basically, performes only one "tuning step" through all the alphas; should be executed multiple times
- **RETURNS** - a tuple with the predictions and an accuracy respectively.

## Getting Support Vectors
```python
def get_support_vectors(zero: float = 10e-5) -> ndarray
```
- **zero** - a number that is being compared with all alphas as those that don't belong to support vectors may not converge to zero completely.
- **RETURNS** - a numpy array of the support vectors from the training dataset.

# Experiments
## The Titanic Dataset
The number of iterations is always **30**, numerical tolerance is always the same: **10e-4**. A random seed for numpy (to generate alphas) is **42**. You can find <a href="https://www.kaggle.com/competitions/titanic">the dataset</a> on Kaggle.

| № | Kernel | C | Gamma | Training Accuracy | Testing Accuracy |
| - | ------ | - | ----- | ----------------- | ---------------- |
| 1 | Linear | 0.01 | - | 72.13% | 61.03% |
| 2 | Linear | 0.1 | - | 69.47% | 56.19% |
| 3 | Linear | 1 | - | 66.81% | 56.49% |
| 4 | Linear | 10 | - | 64.99% | 56.19% |
| 5 | Linear | 100 | - | 65.12% | 56.19% |
| 6 | RBF | 0.01 | 0.1 | 59.38% | 61.63% |
| 7 | RBF | 0.1 | 0.1 | 81.23% | 60.72% |
| 8 | RBF | 1 | 0.1 | 86.41%  | 61.33% |
| 9 | RBF | 10 | 0.1 | 94.82%  | 67.37% |
| 10 | RBF | 100 | 0.1 | 94.54%  | 60.73% |
| 11 | RBF | 10 | 1 | 97.20%  | 53.78% |
| 12 | RBF | 10 | 0.01 | 81.79%  | 77.34% |
