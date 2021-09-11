from numpy import newaxis
from Model import Model
import dill
import numpy as np


class NeuralNetwork:

    def __init__(self, Layers=None):
        self.layers = []
        if Layers != None:
            self.layers = Layers


    def addLayer(self, layer):
        self.layers.append(layer)


    def printWeights(self, bool):
        for layer in self.layers:
            if layer.trainable == True:
                if bool:
                    print(layer.weights)
                else:
                    print(layer.weights.shape)


    def extractWeights(self):
        params = np.array([])
        for layer in self.layers:
            if layer.trainable == True:
                params = np.concatenate((params, np.ravel(layer.weights)))
        return params


    def train(self, train_X, train_Y, Epochs, cost, learning_rate, lmbda, test):
        m = train_Y.shape[1]
        for e in range(Epochs):
            output = train_X
            for layer in self.layers:
                output = layer.forward(output)
            grad = cost.EvaluatePrime(output, train_Y)
            for layer in reversed(self.layers):
                grad = layer.backward(grad, learning_rate, lmbda/m)
            print("Epoch: ", e)
            self.getAccuracy(train_X, test)

    def forward(self, x):
        z = x
        for layer in self.layers:
            z = layer.forward(z)
        return z

    def saveModel(self, name):
        trainedModel = Model(self.layers)
        with (open(name, "wb")) as f:
            dill.dump(trainedModel, f)

    def getAccuracy(self, X, Y):
        z = self.forward(X)
        zA = np.argmax(z, axis=0)
        summation = np.sum(zA == Y)
        print("Accuracy = ", (summation / Y.shape[1]) * 100)
        return z

def loadModel(name):
    with (open(name, "rb")) as file:
        model = dill.load(file)
    return model
