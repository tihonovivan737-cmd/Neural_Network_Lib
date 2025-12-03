import cupy as cp
from layers import LayerWithParameters
from cupy.lib.stride_tricks import as_strided

class AveragePooling2D:
    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.inputs = None
        self.input_shape = None

    def forward(self, inputs):
        assert inputs.ndim == 4, "Вход должен быть 4D (batch_size, height, width, channels)"
        self.input_shape = inputs.shape
        batch_size, height, width, channels = inputs.shape

        assert (height - self.pool_size) % self.stride == 0, "Недопустимая высота для пулинга"
        assert (width - self.pool_size) % self.stride == 0, "Недопустимая ширина для пулинга"

        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1

        self.inputs = inputs

        strides = (
            inputs.strides[0],
            self.stride * inputs.strides[1],
            self.stride * inputs.strides[2],
            inputs.strides[1],
            inputs.strides[2],
            inputs.strides[3]
        )
        view = as_strided(
            inputs,
            shape=(batch_size, output_height, output_width, self.pool_size, self.pool_size, channels),
            strides=strides
        )

        outputs = cp.mean(view, axis=(3, 4))

        assert outputs.shape == (batch_size, output_height, output_width, channels), \
            f"Непредвиденная форма вывода: {outputs.shape}, ожидалось: {(batch_size, output_height, output_width, channels)}"

        return outputs

    def backward(self, d_out):
        batch_size, out_height, out_width, channels = d_out.shape
        assert d_out.shape[0] == self.input_shape[0], "Несоответствие размера батча"
        assert d_out.shape[3] == self.input_shape[3], "Несоответствие количества каналов"

        d_inputs = cp.zeros_like(self.inputs)

        scale = 1.0 / (self.pool_size * self.pool_size)
        d_out_scaled = d_out * scale

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                d_inputs[:, h_start:h_end, w_start:w_end, :] += d_out_scaled[:, i, j, :][:, None, None, :]

        return d_inputs

class MaxPooling2D:
    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.inputs = None
        self.input_shape = None

    def forward(self, inputs):
        assert inputs.ndim == 4, "Вход должен быть 4D (batch_size, height, width, channels)"
        self.input_shape = inputs.shape
        batch_size, height, width, channels = inputs.shape

        assert (height - self.pool_size) % self.stride == 0, "Недопустимая высота для пулинга"
        assert (width - self.pool_size) % self.stride == 0, "Недопустимая ширина для пулинга"

        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1

        self.inputs = inputs

        strides = (
            inputs.strides[0],
            self.stride * inputs.strides[1],
            self.stride * inputs.strides[2],
            inputs.strides[1],
            inputs.strides[2],
            inputs.strides[3]
        )
        view = as_strided(
            inputs,
            shape=(batch_size, output_height, output_width, self.pool_size, self.pool_size, channels),
            strides=strides
        )

        outputs = cp.max(view, axis=(3, 4))

        assert outputs.shape == (batch_size, output_height, output_width, channels), \
            f"Непредвиденная форма вывода: {outputs.shape}, ожидалось: {(batch_size, output_height, output_width, channels)}"

        return outputs

    def backward(self, d_out):
        batch_size, out_height, out_width, channels = d_out.shape
        assert d_out.shape[0] == self.input_shape[0], "Несоответствие размера батча"
        assert d_out.shape[3] == self.input_shape[3], "Несоответствие количества каналов"

        d_inputs = cp.zeros_like(self.inputs)

        scale = 1.0 / (self.pool_size * self.pool_size)
        d_out_scaled = d_out * scale

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                d_inputs[:, h_start:h_end, w_start:w_end, :] += d_out_scaled[:, i, j, :][:, None, None, :]

        return d_inputs

class Conv2D(LayerWithParameters):
    def __init__(self, num_filters, kernel_size, stride=1, padding=0, in_channels=1,
                 weights_initializer='he', biases_initializer='uniform', activation_func=None):
        self.params = {
            'num_filters': num_filters,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'in_channels': in_channels,
        }

        super().__init__(weights_shape=(num_filters, kernel_size, kernel_size, in_channels),
                         biases_shape=(num_filters,), decay_rate=0.9, epsilon=1e-8)
        k = kernel_size
        fan_in = in_channels * k * k
        fan_out = num_filters * k * k
        if weights_initializer == 'he':
            std = cp.sqrt(2.0 / fan_in)
            self.weights = cp.random.randn(num_filters, k, k, in_channels) * std
        elif weights_initializer == 'xavier':
            std = cp.sqrt(2.0 / (fan_in + fan_out))
            self.weights = cp.random.randn(num_filters, k, k, in_channels) * std
        elif weights_initializer == 'random':
            self.weights = cp.random.randn(num_filters, k, k, in_channels) * 0.01
        elif weights_initializer == 'uniform':
            limit = cp.sqrt(6.0 / fan_in)
            self.weights = cp.random.uniform(-limit, limit, (num_filters, k, k, in_channels))
        else:
            raise ValueError(f"Неизвестный инициализатор весов: {weights_initializer}")
        if biases_initializer == 'zeros':
            self.biases = cp.zeros((num_filters,))
        elif biases_initializer == 'ones':
            self.biases = cp.ones((num_filters,))
        elif biases_initializer == 'normal':
            self.biases = cp.random.normal(0, 1, (num_filters,))
        elif biases_initializer == 'uniform':
            limit = cp.sqrt(6.0 / fan_in)
            self.biases = cp.random.uniform(-limit, limit, (num_filters,))
        else:
            raise ValueError(f"Неизвестный инициализатор смещений: {biases_initializer}")
        self.activation_func = activation_func

    def im2col(self, inputs):
        p, s, k = self.params['padding'], self.params['stride'], self.params['kernel_size']
        inputs = cp.asarray(inputs)
        if p > 0:
            inputs = cp.pad(inputs, ((0, 0), (p, p), (p, p), (0, 0)), mode='constant', constant_values=0)
        batch_size, height, width, in_channels = inputs.shape
        out_h = (height - k) // s + 1
        out_w = (width - k) // s + 1

        strides = (inputs.strides[0], s * inputs.strides[1], s * inputs.strides[2],
                   inputs.strides[1], inputs.strides[2], inputs.strides[3])
        strided = cp.lib.stride_tricks.as_strided(inputs,
                                                  shape=(batch_size, out_h, out_w, k, k, in_channels),
                                                  strides=strides)

        col = strided.reshape(batch_size * out_h * out_w, k * k * in_channels)
        return col, out_h, out_w

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, height, width, in_channels = inputs.shape
        assert in_channels == self.params[
            'in_channels'], f"Несоответствие входных каналов: ожидается {self.params['in_channels']}, получено {in_channels}"

        im2col_matrix, out_h, out_w = self.im2col(inputs)

        weights_flat = self.weights.reshape(self.params['num_filters'], -1)

        output = cp.dot(im2col_matrix, weights_flat.T)

        output = output.reshape(batch_size, out_h, out_w, self.params['num_filters'])
        self.z = output + self.biases[None, None, None, :]


        if self.activation_func is not None:
            self.a = self.activation_func(self.z)
            return self.a
        return self.z

    def backward(self, d_out):
        batch_size, out_h, out_w, num_filters = d_out.shape
        im2col_matrix, _, _ = self.im2col(self.inputs)
        if self.activation_func is not None:
            grad_activation = self.activation_func.derivative(self.z)
            d_out = d_out * grad_activation
        d_out_flat = d_out.reshape(batch_size * out_h * out_w, num_filters)

        # Градиент весов
        self.grad_weights = cp.dot(d_out_flat.T, im2col_matrix).reshape(self.weights.shape)
        self.grad_biases = cp.sum(d_out, axis=(0, 1, 2))

        # Градиент входа
        weights_flat = self.weights.reshape(self.params['num_filters'], -1)
        d_inputs_flat = cp.dot(d_out_flat, weights_flat)
        d_inputs = self._col2im(d_inputs_flat, self.inputs.shape)
        return d_inputs

    def _col2im(self, cols, input_shape):
        batch_size, h, w, in_channels = input_shape
        p, s, k = self.params['padding'], self.params['stride'], self.params['kernel_size']
        out_h = (h + 2 * p - k) // s + 1
        out_w = (w + 2 * p - k) // s + 1
        d_inputs = cp.zeros((batch_size, h + 2 * p, w + 2 * p, in_channels), dtype=cp.float32)
        cols_reshaped = cols.reshape(batch_size, out_h, out_w, k, k, in_channels)
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s
                w_start = j * s
                d_inputs[:, h_start:h_start + k, w_start:w_start + k, :] += cols_reshaped[:, i, j, :, :, :]
        if p > 0:
            d_inputs = d_inputs[:, p:-p, p:-p, :]
        return d_inputs


class BatchNorm2D(LayerWithParameters):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, biases_initializer='zeros'):
        super().__init__(
            weights_shape=(num_features,),  # gamma
            biases_shape=(num_features,),  # beta
            decay_rate=0.9, epsilon=eps
        )
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weights = cp.ones(num_features, dtype=cp.float32)  # gamma
        if biases_initializer == 'zeros':
            self.biases = cp.zeros(num_features, dtype=cp.float32)  # beta
        else:
            raise ValueError(f"Неподдерживаемый инициализатор смещений: {biases_initializer}")
        self.running_mean = cp.zeros(num_features, dtype=cp.float32)
        self.running_var = cp.ones(num_features, dtype=cp.float32)
        self.cache = None

    def forward(self, inputs, training=True):
        batch_size, height, width, channels = inputs.shape
        assert channels == self.num_features, f"Ожидается {self.num_features} каналов, получено {channels}"

        if training and batch_size > 1:
            mu = cp.mean(inputs, axis=(0, 1, 2), keepdims=False)
            var = cp.var(inputs, axis=(0, 1, 2), keepdims=False)

            if cp.any(cp.isnan(mu)) or cp.any(cp.isnan(var)):
                raise ValueError("Обнаружен NaN в среднем значении или дисперсии")
            if cp.any(var < 0):
                var = cp.maximum(var, 0)

            inputs_hat = (inputs - mu[None, None, None, :]) / cp.sqrt(var[None, None, None, :] + self.eps)
            self.cache = (inputs, inputs_hat, mu, var)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            inputs_hat = (inputs - self.running_mean[None, None, None, :]) / cp.sqrt(
                self.running_var[None, None, None, :] + self.eps)

        output = self.weights[None, None, None, :] * inputs_hat + self.biases[None, None, None, :]
        return output

    def backward(self, d_out):
        if self.cache is None:
            raise ValueError("Должен быть вызван прямой проход с training=True перед обратным")

        inputs, inputs_hat, mu, var = self.cache
        batch_size, height, width, channels = inputs.shape
        N = batch_size * height * width

        self.grad_weights = cp.sum(d_out * inputs_hat, axis=(0, 1, 2))
        self.grad_biases = cp.sum(d_out, axis=(0, 1, 2))

        d_inputs_hat = d_out * self.weights[None, None, None, :]

        var_eps = var + self.eps
        d_var = cp.sum(
            d_inputs_hat * (inputs - mu[None, None, None, :]) * -0.5 * (var_eps[None, None, None, :] ** -1.5),
            axis=(0, 1, 2))
        d_mu = cp.sum(d_inputs_hat * (-1 / cp.sqrt(var_eps[None, None, None, :])), axis=(0, 1, 2)) + \
               d_var * cp.sum(-2 * (inputs - mu[None, None, None, :]), axis=(0, 1, 2)) / N

        d_inputs = (d_inputs_hat / cp.sqrt(var_eps[None, None, None, :]) +
                    d_var[None, None, None, :] * 2 * (inputs - mu[None, None, None, :]) / N +
                    d_mu[None, None, None, :] / N)

        return d_inputs