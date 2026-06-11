import math
import numpy as np


def calculate_entropy(y):
    """
    Calulate the entropy of label array y
    """

    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)