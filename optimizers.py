import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def step(self, model):
        raise NotImplementedError

    def zero_grad(self, model):
        for layer in model.layers:
            layer.zero_grad()

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, weight_decay=0):
        super().__init__(learning_rate)
        self.weight_decay = weight_decay

    def step(self, model):
        for layer in model.layers:
            for name, param, grad in layer.parameters():
                if self.weight_decay != 0:
                    grad += self.weight_decay * param
                param -= self.lr * grad

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}  # будет хранить скорости для каждого параметра

    def step(self, model):
        for layer_idx, layer in enumerate(model.layers):
            for name, param, grad in layer.parameters():
                key = (layer_idx, name)
                if key not in self.velocities:
                    self.velocities[key] = np.zeros_like(param)
                if self.weight_decay != 0:
                    grad += self.weight_decay * param
                self.velocities[key] = self.momentum * self.velocities[key] + self.lr * grad
                param -= self.velocities[key]

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, model):
        self.t += 1
        for layer_idx, layer in enumerate(model.layers):
            for name, param, grad in layer.parameters():
                key = (layer_idx, name)
                if key not in self.m:
                    self.m[key] = np.zeros_like(param)
                    self.v[key] = np.zeros_like(param)
                if self.weight_decay != 0:
                    grad += self.weight_decay * param
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
