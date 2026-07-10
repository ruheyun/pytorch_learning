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

    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features):
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])

        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)
            mse = np.mean(0.5 * (y - y_pred) ** 2 + self.regularization(self.w))
            self.training_errors.append(mse)
            grad_w = -(y - y_pred).dot(X) / X.shape[0] + self.regularization.grad(self.w)
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred
    

class LinearRegression(Regression):
    """
    Linear model.
    Parameters:
        n_iterations: float. The number of training iterations the algorithm will tune the weights for.
        learning_rate: float. The step length that will be used when updating the weights.
        gradient_descent: boolean. True or false depending if gradient descent should be used when training.
                          If false then we use batch optimization by least squares.
    """

    def __int__(self, n_iterations=100, learning_rate=1e-3, gradient_descent=True):
        super().__init__(n_iterations=n_iterations, learning_rate=learning_rate)
        self.gradient_descent = gradient_descent
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0

    def fit(self, X, y):
        if not self.gradient_descent:
            X = np.insert(X, 0, 1, axis=1)
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            super().fit(X, y)


class LassoRegression(Regression):
    """
    Linear regression model with a regularization factor which does both variable selection
    and regularization. Model that tries to balance the fit of the model with respect to the
    training data and the complexity of the model. A large regularization factor with decreases
    the variance of the model and do para.
    Parameters:
    degree: int. The degree of the polynomial that the independent variable X will be transformed to.
    reg_factor: float. The factor that will determine the amount of regularization and feature shrinkage.
    n_iterations: float. The number of training iterations the algorithm will tune the weights for.
    learning_rate: float. The step length that will be used when updating the weights.
    """

    def __init__(self, degree, reg_factor, n_iterations, learning_rate):
        super().__init__(n_iterations, learning_rate)
        self.degree = degree
        self.regularization = l1_regularization(alpha=reg_factor)
        
    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super().fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super().predict(X)
    

class PolynomialRegression(Regression):
    """
    Performs a non-linear transformation of the data before fitting the model and doing
    predictions which allows for doing non-linear regression.
    Parameters:
    degree: int. The degree of the polynominal that the independent variable X will be transformed to.
    n_iterations: float. The number of training iterations the algorithm will tune the weights for.
    learning_rate: float. The step length that will be used when updating the weights.
    """

    def __init__(self, degree, n_iterations, learning_rate):
        super().__init__(n_iterations, learning_rate)
        self.degree = degree
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0

    def fit(self, X, y):
        X = polynomial_features(X, degree=self.degree)
        super().fit(X, y)

    def predict(self, X):
        X = polynomial_features(X, degree=self.degree)
        return super().predict(X)
    

class RidgeRegression(Regression):
    """
    Parameters:
    -----------
    reg_factor: float
        The factor that will determine the amount of regularization and feature shrinkage.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """

    def __int__(self, reg_factor, n_iterations=1000, learning_rate=0.001):
        self.regularization = l2_regularization(alpha=reg_factor)
        super(RidgeRegression, self).__init__(n_iterations, learning_rate)


class PolynomialRidgeRegression(Regression):
    """
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    reg_factor: float
        The factor that will determine the amount of regularization and feature shrinkage.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """

    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01, gradient_descent=True):
        self.degree = degree
        self.regularization = l2_regularization(alpha=reg_factor)
        super().__init__(n_iterations, learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super().fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super().predict(X)
    

class ElasticNet(Regression):
    """
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    reg_factor: float
        The factor that will determine the amount of regularization and feature shrinkage.
    l1_ration: float
        Weights the contribution of l1 and l2 regularization.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """

    def __int__(self, degree=1, reg_factor=0.05, l1_ratio=0.5, n_iterations=3000, learning_rate=0.01):
        self.degree = degree
        self.regularization = l1_l2_regularization(alpha=reg_factor, l1_ratio=l1_ratio)
        super().__init__(n_iterations, learning_rate)
    
    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super().fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super().predict(X)
 