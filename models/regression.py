import numpy as np
import math
from utils import normalize, polynomial_features


class l1_regularization():
    """
    Regularization for Lasso Regression
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w, ord=1)
    
    def grad(self, w):
        return self.alpha * np.sign(w)
    

class l2_regularization():
    """
    Regularization for Ridge Regression
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * 0.5 * w.T.dot(w)
    
    def grad(self, w):
        return self.alpha * w
    

class l1_l2_regularization():
    """
    Regularization for Elastic Net Regression
    """

    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contr = self.l1_ratio * np.linalg.norm(w)
        l2_contr = (1- self.l1_ratio) * 0.5 * w.T.dot(w)
        return self.alpha * (l1_contr + l2_contr)

    def grad(self, w):
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr)


class Regression(object):
    """
    Base regression model. Models the relationship between a scalar depend
    variable y and the independent variables X.

    Parameters:
        n_iterations: float. The number of training iterations the algorithm will tune the weights for.
        learning_rate: float. The step length that will be used when updating the weights.
    """




    