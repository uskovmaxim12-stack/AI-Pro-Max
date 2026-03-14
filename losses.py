import numpy as np

class Loss:
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class MSELoss(Loss):
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_pred.shape[0]

class CrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        """
        y_pred: (batch_size, num_classes) - логиты (до softmax)
        y_true: (batch_size,) - индексы классов
        """
        self.y_pred = y_pred
        self.y_true = y_true
        # Стабильный softmax
        exps = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        self.softmax = exps / np.sum(exps, axis=1, keepdims=True)
        batch_size = y_pred.shape[0]
        correct_logprobs = -np.log(self.softmax[range(batch_size), y_true] + 1e-10)
        return np.mean(correct_logprobs)

    def backward(self):
        batch_size = self.y_pred.shape[0]
        grad = self.softmax.copy()
        grad[range(batch_size), self.y_true] -= 1
        grad /= batch_size
        return grad
