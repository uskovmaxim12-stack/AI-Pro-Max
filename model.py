import numpy as np
from .layers import Layer

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, training=True):
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                x = layer.forward(x, training) if layer.__class__.__name__ == 'Dropout' else layer.forward(x)
            else:
                x = layer(x)  # активации
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad)
        return grad

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'trainable'):
                layer.training = True

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'trainable'):
                layer.training = False
