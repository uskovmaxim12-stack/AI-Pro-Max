import numpy as np

class Activation:
    """Базовый класс для функций активации."""
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

class ReLU(Activation):
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)

class Sigmoid(Activation):
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)

class Tanh(Activation):
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)

class Softmax(Activation):
    def forward(self, x):
        # Стабильный softmax
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        # Градиент softmax обычно используется с cross-entropy, поэтому здесь упрощённо
        # Для самостоятельного использования нужно реализовать полный градиент, но мы будем использовать в паре с CrossEntropyLoss
        return grad_output  # заглушка
