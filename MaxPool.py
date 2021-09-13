import numpy as np
from skimage.measure import block_reduce
from Layer import Layer

import numpy as np

def MaxPooling(input, f):
    indices = [ [] for _ in range(input.shape[0])]
    max = np.zeros((input.shape[0],input.shape[1] - f + 1, input.shape[1] - f + 1))
    a = 0
    b = 0
    i = 0
    j = 0
    for z in range(input.shape[0]):
        while i + f <= input.shape[1]:
            while j + f <= input.shape[1]:
                temp = input[z, i:(f + i), j:(f + j)]
                indices[z].append(np.array(np.unravel_index(temp.argmax(), temp.shape)) + np.array([i, j]))
                max[z, a, b] = np.max(temp)
                b += 1
                j += 1
            b = 0
            j = 0
            a += 1
            i += 1
        i = 0
        j = 0
        b = 0
        a = 0
    return max, np.asarray(indices)


class MaxPool():

    def __init__(self, filterSize):
        self.trainable = False
        self.size = filterSize
        self.mask = None
        self.indices = None

    def forward(self, input):
        self.mask = np.zeros(input.shape)
        toReturn, self.indices = MaxPooling(input, self.size)
        return toReturn

    def backward(self, output, learning_rate, lmbda):
        a = 0
        for z in range(output.shape[0]):
          for i in range(output.shape[1]):
              for j in range(output.shape[1]):
                  self.mask[(z,) + tuple(self.indices[z, a, :])] = output[z, i, j]
                  a += 1
          a = 0
          i = 0
          j = 0
        return self.mask
