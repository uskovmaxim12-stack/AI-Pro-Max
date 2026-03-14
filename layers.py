import numpy as np
from .activations import Activation

class Layer:
    """Базовый класс для всех слоёв."""
    def __init__(self):
        self.trainable = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def parameters(self):
        return []

    def zero_grad(self):
        pass

class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True, initialization='xavier'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Инициализация весов
        if initialization == 'xavier':
            limit = np.sqrt(6 / (in_features + out_features))
            self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        elif initialization == 'he':
            self.W = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        else:
            self.W = np.random.randn(in_features, out_features) * 0.01

        self.b = np.zeros((1, out_features)) if bias else None

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None

    def forward(self, x):
        self.x = x
        out = x @ self.W
        if self.use_bias:
            out += self.b
        return out

    def backward(self, grad_output):
        # grad_output: (batch_size, out_features)
        self.dW = self.x.T @ grad_output
        if self.use_bias:
            self.db = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = grad_output @ self.W.T
        return grad_input

    def parameters(self):
        params = [('W', self.W, self.dW)]
        if self.use_bias:
            params.append(('b', self.b, self.db))
        return params

    def zero_grad(self):
        self.dW.fill(0)
        if self.use_bias:
            self.db.fill(0)

class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.trainable = False
        self.mask = None

    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        else:
            return x

    def backward(self, grad_output):
        return grad_output * self.mask
