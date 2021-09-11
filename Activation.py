from Layer import Layer
import numpy as np


class Activation(Layer):

    def __init__(self, activationLambda, activationPrimeLambda):
        self.activation = activationLambda
        self.activationPrime = activationPrimeLambda
        self.trainable = False

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, output_gradient, learning_rate, lmdaM):
        return np.multiply(output_gradient, self.activationPrime(self.input))
