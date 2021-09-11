from Activation import Activation
import numpy as np
from scipy.special import expit
from functools import wraps

class Tanh(Activation):

    def __init__(self):
        def tanh_prime(x):
            return 1 - np.power(np.tanh(x), 2)
        super().__init__(np.tanh, tanh_prime)


class Sigmoid(Activation):

    def __init__(self):
        def sigmoid_prime(x):
            val = expit(x)
            return np.multiply(val, 1 - val)
        super().__init__(expit, sigmoid_prime)

class ReLU(Activation):
    def __init__(self):
        def ReLU_Fun(z):
            return np.maximum(0, z, dtype="float32")

        def ReLU_Prime(z):
            return np.asarray(z > 0, dtype="float32")
        super().__init__(ReLU_Fun, ReLU_Prime)
