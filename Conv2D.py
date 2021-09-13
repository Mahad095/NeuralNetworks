from Layer import Layer
import numpy as np
from scipy.signal import correlate, correlate2d
from MNIST import showImage
# conv2D expects input data that is in the order num of images x width x height. So basically, a three dimensional np.array

class Conv2D(Layer):
    def __init__(self, size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.randn(size, size) # n is number of filters, k is the no of channels in the input
        self.bias = np.random.randn(1)
        self.s = size

    def assignParams(self, f, b):
        self.weights = f
        self.s = f.shape[0]
        self.bias = b

    def forward(self, input):
        self.input = input
        temp = np.zeros((input.shape[0], input.shape[1] - self.s + 1, input.shape[1] - self.s + 1))
        for i in range(input.shape[0]):
            temp[i, :, :] = correlate2d(input[i, :, :], self.weights, 'valid')
        showImage(temp[0, :, :], (temp.shape[1], temp.shape[2]))
        return temp + self.bias

    def backward(self, output, learning_rate, lambdM):
        weight_gradients = correlate(self.input, output, 'valid')
        weight_gradients = np.reshape(weight_gradients, (weight_gradients.shape[1], weight_gradients.shape[1]))
        retVal = self.input
        tempFilter = np.fliplr(np.flipud(self.weights))
        for i in range(self.input.shape[0]):
            retVal[i, :, :] = correlate2d(tempFilter, output[i, :, :], 'full')
        self.weights -= learning_rate * weight_gradients + lambdM * np.sum(self.weights)
        self.bias -= learning_rate * np.sum(output)
        return retVal


