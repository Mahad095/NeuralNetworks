from Layer import Layer
import numpy as np
class Reshape(Layer):

    def __init__(self, OutputShape, InputShape):
        super().__init__()
        self.trainable = False
        self.outshape = OutputShape
        self.inshape = InputShape

    def forward(self, input):
        self.input = input
        return np.reshape(input, (self.outshape[1], self.outshape[0])).T

    def backward(self, output, learning_rate, lmbda):
        return output.T.reshape(self.inshape)
