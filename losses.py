import cupy as cp


class MeanSquaredError:
    def __call__(self, y_true, y_pred):
        return cp.mean((y_true - y_pred) ** 2)  # Без .get()

    def gradient(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size


class CrossEntropyLoss:
    def __call__(self, y_true, y_pred):
        y_pred = cp.clip(y_pred, 1e-7, 1 - 1e-7)
        batch_size = y_true.shape[0]
        return -cp.sum(y_true * cp.log(y_pred)) / batch_size

    def gradient(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        return (y_pred - y_true) / batch_size


class HingeLoss:
    def __call__(self, y_true, y_pred):
        return cp.mean(cp.maximum(0, 1 - y_true * y_pred))

    def gradient(self, y_true, y_pred):
        return cp.where(y_true * y_pred < 1, -y_true, 0)
