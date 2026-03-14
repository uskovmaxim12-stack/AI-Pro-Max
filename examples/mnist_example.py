import sys
sys.path.append('..')  # чтобы видеть модули выше

import numpy as np
from my_neural_network.layers import Linear, Dropout
from my_neural_network.activations import ReLU, Softmax
from my_neural_network.losses import CrossEntropyLoss
from my_neural_network.optimizers import Adam
from my_neural_network.model import Sequential
from my_neural_network.train import train
from my_neural_network.utils import load_mnist, train_test_split, to_categorical

def main():
    # Загружаем MNIST
    X, y = load_mnist(flatten=True, normalize=True)  # X: (70000, 784), y: (70000,)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Строим модель
    model = Sequential([
        Linear(784, 256, initialization='he'),
        ReLU(),
        Dropout(0.2),
        Linear(256, 128, initialization='he'),
        ReLU(),
        Dropout(0.2),
        Linear(128, 10, initialization='xavier'),
        Softmax()  # на выходе softmax для вероятностей
    ])

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(learning_rate=0.001)

    # Обучаем
    train_losses, val_losses = train(
        model, loss_fn, optimizer,
        X_train, y_train, epochs=10, batch_size=64,
        X_val=X_test, y_val=y_test, verbose=True
    )

    # Оценка точности
    y_pred = model.forward(X_test, training=False)
    predicted_classes = np.argmax(y_pred, axis=1)
    accuracy = np.mean(predicted_classes == y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
