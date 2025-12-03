import cupy as cp


def train(network, x_train, y_train, loss_func, batch_size=None, bt_vs_size=None, evaluate_only=False):
    n_samples = x_train.shape[0]
    total_loss = cp.array(0.0)
    correct_predictions = cp.array(0)
    if batch_size is None:
        batch_size = n_samples
    indices = cp.arange(n_samples)
    cp.random.shuffle(indices)
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        x_batch = x_train_shuffled[start_idx:end_idx]
        y_batch = y_train_shuffled[start_idx:end_idx]
        predictions = network.forward(x_batch, training=True)
        batch_loss = loss_func(y_batch, predictions)
        total_loss += batch_loss * x_batch.shape[0]
        network.backward(y_batch, loss_func)
        network.update_weights()

        predicted_class = cp.argmax(predictions, axis=1)
        true_class = cp.argmax(y_batch, axis=1)
        correct_predictions += cp.sum(predicted_class == true_class)

        if bt_vs_size is not None and (start_idx // batch_size + 1) % bt_vs_size == 0 and not evaluate_only:
            print(f"  Батч {start_idx // batch_size + 1}, Потеря: {batch_loss.get():.4f}")

    epoch_loss = total_loss.get() / n_samples
    accuracy = (correct_predictions / n_samples).get()
    return accuracy, epoch_loss


def evaluate(network, x_data, y_data, loss_func, batch_size=1):
    n_samples = x_data.shape[0]
    total_loss = cp.array(0.0)
    correct_predictions = cp.array(0)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        x_batch = x_data[start_idx:end_idx]
        y_batch = y_data[start_idx:end_idx]
        predictions = network.forward(x_batch, training=False)
        batch_loss = loss_func(y_batch, predictions)
        total_loss += batch_loss * x_batch.shape[0]
        predicted_class = cp.argmax(predictions, axis=1)
        true_class = cp.argmax(y_batch, axis=1)
        correct_predictions += cp.sum(predicted_class == true_class)

    avg_loss = total_loss.get() / n_samples
    accuracy = (correct_predictions / n_samples).get()
    return accuracy, avg_loss
