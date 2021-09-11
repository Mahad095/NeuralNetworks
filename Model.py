from numpy import newaxis

class Model:
    def __init__(self, Trained):
        self.Layers = Trained

    def forward(self, x):
        z = x
        if z.ndim == 1:
            z = z[:, newaxis]
        for layer in self.Layers:
            z = layer.forward(z)
        return z
