import numpy as np


def linear_kernel(**kwargs):
    """
    Linear kernel function 
    """

    def f(x_1, x_2):
        return np.inner(x_1, x_2)
    
    return f


def polynomial_kernel(power, coef, **kwargs):
    """
    Polynomial kernel function
    """

    def f(x_1, x_2):
        return (np.inner(x_1, x_2) + coef) ** power
    
    return f


def rbf_kernel(gamma, **kwargs):
    """
    Radial basis function
    """

    def f(x_1, x_2):
        distance = np.linalg.norm(x_1 - x_2) ** 2
        return np.exp(-gamma * distance)
    
    return f
