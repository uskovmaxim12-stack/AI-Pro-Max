import numpy as np
from tqdm import tqdm  # для прогресс-бара (можно убрать)

def train(model, loss_fn, optimizer, X_train, y_train, epochs=10, batch_size=32, X_val=None, y_val=None, verbose=True):
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        # Перемешивание данных
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0
        num_batches = int(np.ceil(X_train.shape[0] / batch_size))

        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, X_train.shape[0])
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            # Forward
            y_pred = model.forward(X_batch, training=True)
            loss = loss_fn.forward(y_pred, y_batch)
            epoch_loss += loss

            # Backward
            grad = loss_fn.backward()
            model.backward(grad)

            # Update weights
            optimizer.step(model)
            optimizer.zero_grad(model)

        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)

        # Валидация
        if X_val is not None and y_val is not None:
            y_val_pred = model.forward(X_val, training=False)
            val_loss = loss_fn.forward(y_val_pred, y_val)
            val_losses.append(val_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - train loss: {avg_train_loss:.4f} - val loss: {val_loss:.4f}")
        else:
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - train loss: {avg_train_loss:.4f}")

    return train_losses, val_losses
