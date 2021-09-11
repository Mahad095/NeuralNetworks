from Cost import Cost
import numpy as np

class MeanSquaredError(Cost):

    def __init__(self):
        def func(prediction, Actual):
            return (1/2) * np.mean(np.power(prediction - Actual, 2))

        def prime(prediction, Actual):
            return (prediction - Actual)/Actual.shape[1]
        super().__init__(func, prime)

class CrossEntropy(Cost):

    def __init__(self):
        def func(hx, y):
            return (-1/y.shape[1]) * np.sum(np.multiply(y, np.log(hx)) + np.multiply((1 + y), np.log(1-hx)))

        def prime(hx, y):
            return (1/y.shape[1]) * (np.divide(-y, hx) + np.divide((1 - y), (1 - hx)))
        super().__init__(func, prime)