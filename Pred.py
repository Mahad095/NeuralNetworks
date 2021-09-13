from NeuralNetwork import loadModel
from MNIST import load, to_categorical
from MNIST import showImage
import numpy as np
train_X, train_Y, test_X, test_Y = load()
from cv2 import waitKey
test_X = test_X.T

y = test_Y[np.newaxis, :]
nn = loadModel("Model1.py")
z = nn.forward(test_X)
zA = np.argmax(z, axis=0)
for i in range(2000):
    print(zA[i])
    showImage(test_X[:, i])
    waitKey()


