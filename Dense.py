import numpy as np
from Layer import Layer

class Dense(Layer):
    def __init__(self, inputLayerSize, OutputLayerSize):
        super().__init__()
        self.weights = np.random.randn(OutputLayerSize, inputLayerSize)
        self. bias = np.random.randn(OutputLayerSize, 1)

    def forward(self, input):
        self.input = input
        return (self.weights @ input) + self.bias

    def backward(self, output_gradient, learning_rate, lmbdaM):
        weight_gradient = output_gradient @ self.input.T
        retval = self.weights.T @ output_gradient
        self.weights -= (learning_rate * weight_gradient) + lmbdaM * np.sum(self.weights)
        self.bias -= learning_rate * np.sum(output_gradient, axis=1, keepdims=True)
        return retval

