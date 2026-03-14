import numpy as np

def to_categorical(y, num_classes=None):
    """Преобразует метки в one-hot encoding."""
    if num_classes is None:
        num_classes = np.max(y) + 1
    return np.eye(num_classes)[y]

def load_mnist(flatten=True, normalize=True):
    """Загружает MNIST (если есть локально, иначе скачивает). Упрощённо."""
    # В реальности нужно скачать с http://yann.lecun.com/exdb/mnist/
    # Здесь заглушка для примера
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data.astype('float32'), mnist.target.astype('int')
    if normalize:
        X /= 255.0
    if flatten:
        X = X.reshape(X.shape[0], -1)
    return X, y

def train_test_split(X, y, test_size=0.2, shuffle=True):
    indices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    split = int(X.shape[0] * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
