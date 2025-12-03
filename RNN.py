import cupy
import cupy as cp
import cupyx
from activations import Softmax, Sigmoid, Tanh
from layers import LayerWithParameters
from cupy.linalg import svd

def orthogonal_init(shape):
    a = cp.random.randn(*shape)
    u, _, v = svd(a, full_matrices=False)
    return u if u.shape == shape else v

class RecurrentBase(LayerWithParameters):
    def __init__(self, weights_shape, biases_shape, decay_rate=0.9, epsilon=1e-8, beta2=0.999):
        super().__init__(weights_shape=weights_shape, biases_shape=biases_shape, decay_rate=decay_rate, epsilon=epsilon)
        self.beta1 = decay_rate
        self.beta2 = beta2
        self.t = 1

    def _init_layer_params(self, W, b):
        self.weights.append(W)
        self.biases.append(b)
        self.m_weights.append(cp.zeros_like(W))
        self.v_weights.append(cp.zeros_like(W))
        self.m_biases.append(cp.zeros_like(b))
        self.v_biases.append(cp.zeros_like(b))
        self.squared_grad_weights.append(cp.zeros_like(W))
        self.squared_grad_biases.append(cp.zeros_like(b))
        self.grad_weights.append(cp.zeros_like(W))
        self.grad_biases.append(cp.zeros_like(b))

    def update_weights(self, learning_rate=0.001, optimizer="Adam", weight_decay=0):
        clip_value = 5
        for layer_idx in range(self.num_layers):
            norm_weights = cp.sqrt(cp.sum(self.grad_weights[layer_idx] ** 2))
            norm_biases = cp.sqrt(cp.sum(self.grad_biases[layer_idx] ** 2))
            if norm_weights > clip_value:
                self.grad_weights[layer_idx] *= clip_value / norm_weights
            if norm_biases > clip_value:
                self.grad_biases[layer_idx] *= clip_value / norm_biases
            gw = self.grad_weights[layer_idx]
            gb = self.grad_biases[layer_idx]
            if optimizer == "RMSProp":
                self.squared_grad_weights[layer_idx] = self.decay_rate * self.squared_grad_weights[layer_idx] + (1 - self.decay_rate) * (gw ** 2)
                self.squared_grad_biases[layer_idx] = self.decay_rate * self.squared_grad_biases[layer_idx] + (1 - self.decay_rate) * (gb ** 2)
                self.weights[layer_idx] -= learning_rate * gw / (cp.sqrt(self.squared_grad_weights[layer_idx]) + self.epsilon)
                self.biases[layer_idx] -= learning_rate * gb / (cp.sqrt(self.squared_grad_biases[layer_idx]) + self.epsilon)
            elif optimizer == "AdaGrad":
                self.squared_grad_weights[layer_idx] += gw ** 2
                self.squared_grad_biases[layer_idx] += gb ** 2
                self.weights[layer_idx] -= learning_rate * gw / (cp.sqrt(self.squared_grad_weights[layer_idx]) + self.epsilon)
                self.biases[layer_idx] -= learning_rate * gb / (cp.sqrt(self.squared_grad_biases[layer_idx]) + self.epsilon)
            elif optimizer == "Adam":
                gw += weight_decay * self.weights[layer_idx]
                gb += weight_decay * self.biases[layer_idx]
                self.m_weights[layer_idx] = self.beta1 * self.m_weights[layer_idx] + (1 - self.beta1) * gw
                self.v_weights[layer_idx] = self.beta2 * self.v_weights[layer_idx] + (1 - self.beta2) * (gw ** 2)
                self.m_biases[layer_idx] = self.beta1 * self.m_biases[layer_idx] + (1 - self.beta1) * gb
                self.v_biases[layer_idx] = self.beta2 * self.v_biases[layer_idx] + (1 - self.beta2) * (gb ** 2)
                m_weights_hat = self.m_weights[layer_idx] / (1 - self.beta1 ** self.t)
                v_weights_hat = self.v_weights[layer_idx] / (1 - self.beta2 ** self.t)
                m_biases_hat = self.m_biases[layer_idx] / (1 - self.beta1 ** self.t)
                v_biases_hat = self.v_biases[layer_idx] / (1 - self.beta2 ** self.t)
                self.weights[layer_idx] -= learning_rate * m_weights_hat / (cp.sqrt(v_weights_hat) + self.epsilon)
                self.biases[layer_idx] -= learning_rate * m_biases_hat / (cp.sqrt(v_biases_hat) + self.epsilon)
            elif optimizer == "AdaDelta":
                self.squared_grad_weights[layer_idx] = self.decay_rate * self.squared_grad_weights[layer_idx] + (1 - self.decay_rate) * (gw ** 2)
                self.squared_grad_biases[layer_idx] = self.decay_rate * self.squared_grad_biases[layer_idx] + (1 - self.decay_rate) * (gb ** 2)
                update_weights = -gw * (cp.sqrt(self.m_weights[layer_idx] + self.epsilon) / cp.sqrt(self.squared_grad_weights[layer_idx] + self.epsilon))
                update_biases = -gb * (cp.sqrt(self.m_biases[layer_idx] + self.epsilon) / cp.sqrt(self.squared_grad_biases[layer_idx] + self.epsilon))
                self.m_weights[layer_idx] = self.decay_rate * self.m_weights[layer_idx] + (1 - self.decay_rate) * (update_weights ** 2)
                self.m_biases[layer_idx] = self.decay_rate * self.m_biases[layer_idx] + (1 - self.decay_rate) * (update_biases ** 2)
                self.weights[layer_idx] += update_weights
                self.biases[layer_idx] += update_biases
            elif optimizer == "SGD":
                self.weights[layer_idx] -= learning_rate * gw
                self.biases[layer_idx] -= learning_rate * gb
        if optimizer == "Adam":
            self.t += 1
        for layer_idx in range(self.num_layers):
            self.grad_weights[layer_idx][...] = 0
            self.grad_biases[layer_idx][...] = 0

class Embedding(LayerWithParameters):
    def __init__(self, num_embeddings, embedding_dim, weights_initializer='uniform', padding_idx=None):
        self.params = {'num_embeddings': num_embeddings, 'embedding_dim': embedding_dim, 'padding_idx': padding_idx}
        super().__init__(weights_shape=(num_embeddings, embedding_dim), biases_shape=(0,), decay_rate=1, epsilon=1e-8)
        fan_in = embedding_dim
        if weights_initializer == 'he':
            std = cp.sqrt(2.0 / fan_in)
            self.weights = cp.random.randn(num_embeddings, embedding_dim) * std
        elif weights_initializer == 'xavier':
            fan_out = embedding_dim
            std = cp.sqrt(2.0 / (fan_in + fan_out))
            self.weights = cp.random.randn(num_embeddings, embedding_dim) * std
        elif weights_initializer == 'random':
            self.weights = cp.random.randn(num_embeddings, embedding_dim) * 0.01
        elif weights_initializer == 'uniform':
            limit = cp.sqrt(6.0 / fan_in)
            self.weights = cp.random.uniform(-limit, limit, (num_embeddings, embedding_dim))
        else:
            raise ValueError(f"Неизвестный инициализатор весов: {weights_initializer}")
        self.biases = cp.zeros((0,))
        if padding_idx is not None:
            self.weights[padding_idx] = 0

    def forward(self, inputs):
        self.inputs = cp.asarray(inputs)
        assert self.inputs.ndim == 2, "Вход должен быть 2D (batch_size, sequence_length)"
        assert cp.all((self.inputs >= 0) & (self.inputs < self.params['num_embeddings'])), "Индексы вне диапазона"
        outputs = self.weights[self.inputs]
        return outputs

    def backward(self, d_out):
        assert d_out.shape[:2] == self.inputs.shape and d_out.shape[2] == self.params['embedding_dim']
        self.grad_weights = cp.zeros_like(self.weights)
        flat_inputs = self.inputs.ravel()
        flat_d_out = d_out.reshape(-1, self.params['embedding_dim'])
        cupyx.scatter_add(self.grad_weights, (flat_inputs, slice(None)), flat_d_out)
        self.grad_biases = cp.zeros_like(self.biases)
        return cp.zeros(self.inputs.shape, dtype=cp.float32)

class RNN(RecurrentBase):
    def __init__(self, input_size, hidden_size, activation_func, num_layers=1, weights_initializer='xavier', biases_initializer='uniform', decay_rate=0.9, epsilon=1e-8, beta2=0.999):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_func = activation_func
        self.num_layers = num_layers
        super().__init__(weights_shape=(hidden_size, input_size + hidden_size), biases_shape=(1, hidden_size), decay_rate=decay_rate, epsilon=epsilon, beta2=beta2)
        self.weights = []
        self.biases = []
        self.m_weights = []
        self.v_weights = []
        self.m_biases = []
        self.v_biases = []
        self.squared_grad_weights = []
        self.squared_grad_biases = []
        self.grad_weights = []
        self.grad_biases = []
        current_input_size = input_size
        for layer_idx in range(num_layers):
            if weights_initializer == 'random':
                w_ih = cp.random.randn(hidden_size, current_input_size) * 0.01
            elif weights_initializer == 'he':
                std_ih = cp.sqrt(2 / current_input_size)
                w_ih = cp.random.randn(hidden_size, current_input_size) * std_ih
            elif weights_initializer == 'normal':
                w_ih = cp.random.normal(0, 1, (hidden_size, current_input_size))
            elif weights_initializer == 'uniform':
                limit_ih = cp.sqrt(6.0 / current_input_size)
                w_ih = cp.random.uniform(-limit_ih, limit_ih, (hidden_size, current_input_size))
            elif weights_initializer == 'xavier':
                limit_ih = cp.sqrt(6 / (current_input_size + hidden_size))
                w_ih = cp.random.uniform(-limit_ih, limit_ih, (hidden_size, current_input_size))
                limit_hh = cp.sqrt(6 / (hidden_size + hidden_size))
                w_hh = cp.random.uniform(-limit_hh, limit_hh, (hidden_size, hidden_size))
            else:
                raise ValueError(f"Неизвестный инициализатор весов: {weights_initializer}")
            W = cp.concatenate((w_ih, w_hh), axis=1)
            if biases_initializer == 'zeros':
                b = cp.zeros((1, hidden_size))
            elif biases_initializer == 'ones':
                b = cp.ones((1, hidden_size))
            elif biases_initializer == 'normal':
                b = cp.random.normal(0, 1, (1, hidden_size))
            elif biases_initializer == 'uniform':
                limit = cp.sqrt(6.0 / current_input_size)
                b = cp.random.uniform(-limit, limit, (1, hidden_size))
            else:
                raise ValueError(f"Неизвестный инициализатор смещений: {biases_initializer}")
            self._init_layer_params(W, b)
            current_input_size = hidden_size
        self.inputs_per_layer = []
        self.hidden_states_per_layer = []
        self.z_states_per_layer = []

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, seq_len, input_size = inputs.shape
        assert input_size == self.input_size, f"Несоответствие размера входа: ожидалось {self.input_size}, получено {input_size}"
        self.inputs_per_layer = []
        self.hidden_states_per_layer = []
        self.z_states_per_layer = []
        x = inputs
        for layer_idx in range(self.num_layers):
            self.inputs_per_layer.append(cp.copy(x))
            hidden_states = cp.zeros((batch_size, seq_len + 1, self.hidden_size))
            z_states = cp.zeros((batch_size, seq_len, self.hidden_size))
            h_prev = cp.zeros((batch_size, self.hidden_size))
            hidden_states[:, 0, :] = h_prev
            W = self.weights[layer_idx]
            b = self.biases[layer_idx]
            for t in range(seq_len):
                combined = cp.concatenate((x[:, t, :], h_prev), axis=1)
                z = combined @ W.T + b
                z_states[:, t, :] = z
                h = z if self.activation_func is None else self.activation_func(z)
                hidden_states[:, t + 1, :] = h
                h_prev = h
            x = hidden_states[:, 1:, :]
            self.hidden_states_per_layer.append(hidden_states)
            self.z_states_per_layer.append(z_states)
        self.last_hidden = hidden_states[:, -1, :]
        return x

    def backward(self, grad_output):
        batch_size, seq_len, hidden_size = grad_output.shape
        assert hidden_size == self.hidden_size, f"Несоответствие размера скрытого слоя: ожидалось {self.hidden_size}, получено {hidden_size}"
        grad_x = grad_output
        for layer_idx in reversed(range(self.num_layers)):
            x = self.inputs_per_layer[layer_idx]
            hidden_states = self.hidden_states_per_layer[layer_idx]
            z_states = self.z_states_per_layer[layer_idx]
            W = self.weights[layer_idx]
            grad_input_layer = cp.zeros_like(x)
            grad_h_next = cp.zeros((batch_size, self.hidden_size))
            self.grad_weights[layer_idx][...] = 0
            self.grad_biases[layer_idx][...] = 0
            for t in reversed(range(seq_len)):
                grad_h = grad_x[:, t, :] + grad_h_next
                grad_activation = 1.0 if self.activation_func is None else self.activation_func.derivative(z_states[:, t, :])
                grad_z = grad_h if isinstance(self.activation_func, Softmax) else grad_h * grad_activation
                combined = cp.concatenate((x[:, t, :], hidden_states[:, t, :]), axis=1)
                self.grad_weights[layer_idx] += grad_z.T @ combined
                self.grad_biases[layer_idx] += cp.sum(grad_z, axis=0, keepdims=True)
                grad_combined = grad_z @ W
                grad_x_t = grad_combined[:, :x.shape[2]]
                grad_h_prev = grad_combined[:, x.shape[2]:]
                grad_input_layer[:, t, :] = grad_x_t
                grad_h_next = grad_h_prev
            if layer_idx > 0:
                grad_x = grad_input_layer
        return grad_input_layer if self.num_layers > 0 else cp.zeros_like(self.inputs)

class LSTM(RecurrentBase):
    def __init__(self, input_size, hidden_size, activation_func=Tanh, num_layers=1, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', decay_rate=0.9, epsilon=1e-8, beta2=0.999):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_func = activation_func
        self.num_layers = num_layers
        super().__init__(weights_shape=(4 * hidden_size, input_size + hidden_size), biases_shape=(1, 4 * hidden_size), decay_rate=decay_rate, epsilon=epsilon, beta2=beta2)
        self.weights = []
        self.biases = []
        self.m_weights = []
        self.v_weights = []
        self.m_biases = []
        self.v_biases = []
        self.squared_grad_weights = []
        self.squared_grad_biases = []
        self.grad_weights = []
        self.grad_biases = []
        current_input_size = input_size
        for layer_idx in range(num_layers):
            if kernel_initializer == 'glorot_uniform':
                limit_ih = cp.sqrt(6 / (current_input_size + 4 * hidden_size))
                w_ih = cp.random.uniform(-limit_ih, limit_ih, (4 * hidden_size, current_input_size))
            else:
                raise ValueError(f"Unsupported kernel_initializer: {kernel_initializer}")
            if recurrent_initializer == 'orthogonal':
                w_hh = orthogonal_init((4 * hidden_size, hidden_size))
            else:
                raise ValueError(f"Unsupported recurrent_initializer: {recurrent_initializer}")
            W = cp.concatenate((w_ih, w_hh), axis=1)
            if bias_initializer == 'zeros':
                b = cp.zeros((1, 4 * hidden_size))
            else:
                raise ValueError(f"Unsupported bias_initializer: {bias_initializer}")
            self._init_layer_params(W, b)
            current_input_size = hidden_size
        self.inputs_per_layer = []
        self.hidden_states_per_layer = []
        self.cell_states_per_layer = []
        self.gates_per_layer = []

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, seq_len, input_size = inputs.shape
        assert input_size == self.input_size, f"Несоответствие размера входа: ожидалось {self.input_size}, получено {input_size}"
        self.inputs_per_layer = []
        self.hidden_states_per_layer = []
        self.cell_states_per_layer = []
        self.gates_per_layer = []
        x = inputs
        for layer_idx in range(self.num_layers):
            self.inputs_per_layer.append(cp.copy(x))
            hidden_states = cp.zeros((batch_size, seq_len + 1, self.hidden_size))
            cell_states = cp.zeros((batch_size, seq_len + 1, self.hidden_size))
            gates = cp.zeros((batch_size, seq_len, 4 * self.hidden_size))  # i, f, g, o
            h_prev = cp.zeros((batch_size, self.hidden_size))
            c_prev = cp.zeros((batch_size, self.hidden_size))
            hidden_states[:, 0, :] = h_prev
            cell_states[:, 0, :] = c_prev
            W = self.weights[layer_idx]
            b = self.biases[layer_idx]
            for t in range(seq_len):
                combined = cp.concatenate((x[:, t, :], h_prev), axis=1)
                z = combined @ W.T + b
                gates[:, t, :] = z
                i = Sigmoid()(z[:, :self.hidden_size])
                f = Sigmoid()(z[:, self.hidden_size:2*self.hidden_size])
                g = self.activation_func(z[:, 2*self.hidden_size:3*self.hidden_size])
                o = Sigmoid()(z[:, 3*self.hidden_size:])
                c = f * c_prev + i * g
                h = o * self.activation_func(c)
                hidden_states[:, t + 1, :] = h
                cell_states[:, t + 1, :] = c
                h_prev = h
                c_prev = c
            x = hidden_states[:, 1:, :]
            self.hidden_states_per_layer.append(hidden_states)
            self.cell_states_per_layer.append(cell_states)
            self.gates_per_layer.append(gates)
        self.last_hidden = hidden_states[:, -1, :]
        self.last_cell = cell_states[:, -1, :]
        return x

    def backward(self, grad_output):
        batch_size, seq_len, hidden_size = grad_output.shape
        assert hidden_size == self.hidden_size, f"Несоответствие размера скрытого слоя: ожидалось {self.hidden_size}, получено {hidden_size}"
        grad_x = grad_output
        for layer_idx in reversed(range(self.num_layers)):
            x = self.inputs_per_layer[layer_idx]
            hidden_states = self.hidden_states_per_layer[layer_idx]
            cell_states = self.cell_states_per_layer[layer_idx]
            gates = self.gates_per_layer[layer_idx]
            W = self.weights[layer_idx]
            grad_input_layer = cp.zeros_like(x)
            grad_h_next = cp.zeros((batch_size, self.hidden_size))
            grad_c_next = cp.zeros((batch_size, self.hidden_size))
            self.grad_weights[layer_idx][...] = 0
            self.grad_biases[layer_idx][...] = 0
            for t in reversed(range(seq_len)):
                grad_h = grad_x[:, t, :] + grad_h_next
                o = Sigmoid()(gates[:, t, 3*self.hidden_size:])
                c = cell_states[:, t + 1, :]
                tanh_c = self.activation_func(c)
                grad_o = grad_h * tanh_c
                grad_tanh_c = grad_h * o
                grad_c = grad_tanh_c * self.activation_func.derivative(c) + grad_c_next
                i = Sigmoid()(gates[:, t, :self.hidden_size])
                f = Sigmoid()(gates[:, t, self.hidden_size:2*self.hidden_size])
                g = self.activation_func(gates[:, t, 2*self.hidden_size:3*self.hidden_size])
                c_prev = cell_states[:, t, :]
                grad_i = grad_c * g
                grad_f = grad_c * c_prev
                grad_g = grad_c * i
                grad_c_prev = grad_c * f
                grad_z_i = grad_i * Sigmoid.derivative(gates[:, t, :self.hidden_size])
                grad_z_f = grad_f * Sigmoid.derivative(gates[:, t, self.hidden_size:2*self.hidden_size])
                grad_z_g = grad_g * self.activation_func.derivative(gates[:, t, 2*self.hidden_size:3*self.hidden_size])
                grad_z_o = grad_o * Sigmoid.derivative(gates[:, t, 3*self.hidden_size:])
                grad_z = cp.concatenate((grad_z_i, grad_z_f, grad_z_g, grad_z_o), axis=1)
                combined = cp.concatenate((x[:, t, :], hidden_states[:, t, :]), axis=1)
                self.grad_weights[layer_idx] += grad_z.T @ combined
                self.grad_biases[layer_idx] += cp.sum(grad_z, axis=0, keepdims=True)
                grad_combined = grad_z @ W
                grad_x_t = grad_combined[:, :x.shape[2]]
                grad_h_prev = grad_combined[:, x.shape[2]:]
                grad_input_layer[:, t, :] = grad_x_t
                grad_h_next = grad_h_prev
                grad_c_next = grad_c_prev
            if layer_idx > 0:
                grad_x = grad_input_layer
        return grad_input_layer if self.num_layers > 0 else cp.zeros_like(self.inputs)

class minLSTM(RecurrentBase):
    def __init__(self, input_size, hidden_size, num_layers=1, kernel_initializer='glorot_uniform', bias_initializer='zeros', decay_rate=0.9, epsilon=1e-8, beta2=0.999):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        super().__init__(weights_shape=(3 * hidden_size, input_size), biases_shape=(1, 3 * hidden_size), decay_rate=decay_rate, epsilon=epsilon, beta2=beta2)
        self.weights = []
        self.biases = []
        self.m_weights = []
        self.v_weights = []
        self.m_biases = []
        self.v_biases = []
        self.squared_grad_weights = []
        self.squared_grad_biases = []
        self.grad_weights = []
        self.grad_biases = []
        current_input_size = input_size
        for layer_idx in range(num_layers):
            if kernel_initializer == 'glorot_uniform':
                limit_ih = cp.sqrt(6 / (current_input_size + 3 * hidden_size))
                w_ih = cp.random.uniform(-limit_ih, limit_ih, (3 * hidden_size, current_input_size))
            else:
                raise ValueError(f"Unsupported kernel_initializer: {kernel_initializer}")
            W = w_ih
            if bias_initializer == 'zeros':
                b = cp.zeros((1, 3 * hidden_size))
            else:
                raise ValueError(f"Unsupported bias_initializer: {bias_initializer}")
            self._init_layer_params(W, b)
            current_input_size = hidden_size
        self.inputs_per_layer = []
        self.hidden_states_per_layer = []
        self.gates_per_layer = []

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, seq_len, input_size = inputs.shape
        assert input_size == self.input_size, f"Input size mismatch: expected {self.input_size}, got {input_size}"
        self.inputs_per_layer = []
        self.hidden_states_per_layer = []
        self.gates_per_layer = []
        x = inputs
        for layer_idx in range(self.num_layers):
            self.inputs_per_layer.append(cp.copy(x))
            hidden_states = cp.zeros((batch_size, seq_len + 1, self.hidden_size))
            gates = cp.zeros((batch_size, seq_len, 3 * self.hidden_size))  # f, i, n
            h_prev = cp.zeros((batch_size, self.hidden_size))
            hidden_states[:, 0, :] = h_prev
            W = self.weights[layer_idx]
            b = self.biases[layer_idx]
            for t in range(seq_len):
                x_t = x[:, t, :]
                linear = x_t @ W.T + b
                gates[:, t, :] = linear
                f = Sigmoid()(linear[:, :self.hidden_size])
                i = Sigmoid()(linear[:, self.hidden_size:2 * self.hidden_size])
                n = linear[:, 2 * self.hidden_size:]  # No activation
                denom = f + i + self.epsilon
                f_prime = f / denom
                i_prime = i / denom
                h = f_prime * h_prev + i_prime * n
                hidden_states[:, t + 1, :] = h
                h_prev = h
            x = hidden_states[:, 1:, :]
            self.hidden_states_per_layer.append(hidden_states)
            self.gates_per_layer.append(gates)
        self.last_hidden = hidden_states[:, -1, :]
        return x

    def backward(self, grad_output):
        batch_size, seq_len, hidden_size = grad_output.shape
        assert hidden_size == self.hidden_size, f"Hidden size mismatch: expected {self.hidden_size}, got {hidden_size}"
        grad_x = grad_output
        for layer_idx in reversed(range(self.num_layers)):
            x = self.inputs_per_layer[layer_idx]
            hidden_states = self.hidden_states_per_layer[layer_idx]
            gates = self.gates_per_layer[layer_idx]
            W = self.weights[layer_idx]
            grad_input_layer = cp.zeros_like(x)
            grad_h_next = cp.zeros((batch_size, self.hidden_size))
            self.grad_weights[layer_idx][...] = 0
            self.grad_biases[layer_idx][...] = 0
            for t in reversed(range(seq_len)):
                grad_h = grad_x[:, t, :] + grad_h_next
                z_f = gates[:, t, :self.hidden_size]
                z_i = gates[:, t, self.hidden_size:2 * self.hidden_size]
                z_n = gates[:, t, 2 * self.hidden_size:]
                f = Sigmoid()(z_f)
                i = Sigmoid()(z_i)
                n = z_n
                denom = f + i + self.epsilon
                f_prime = f / denom
                i_prime = i / denom
                h_prev = hidden_states[:, t, :]
                grad_f_prime = grad_h * h_prev
                grad_i_prime = grad_h * n
                grad_n = grad_h * i_prime
                grad_h_prev = grad_h * f_prime
                d_square = denom ** 2
                grad_f = (i / d_square) * (grad_f_prime - grad_i_prime)
                grad_i = (f / d_square) * (grad_i_prime - grad_f_prime)
                grad_z_f = grad_f * Sigmoid.derivative(z_f)
                grad_z_i = grad_i * Sigmoid.derivative(z_i)
                grad_z_n = grad_n  # Derivative of identity is 1
                x_t = x[:, t, :]
                self.grad_weights[layer_idx][:self.hidden_size, :] += grad_z_f.T @ x_t
                self.grad_weights[layer_idx][self.hidden_size:2 * self.hidden_size, :] += grad_z_i.T @ x_t
                self.grad_weights[layer_idx][2 * self.hidden_size:, :] += grad_z_n.T @ x_t
                self.grad_biases[layer_idx][:, :self.hidden_size] += cp.sum(grad_z_f, axis=0, keepdims=True)
                self.grad_biases[layer_idx][:, self.hidden_size:2 * self.hidden_size] += cp.sum(grad_z_i, axis=0, keepdims=True)
                self.grad_biases[layer_idx][:, 2 * self.hidden_size:] += cp.sum(grad_z_n, axis=0, keepdims=True)
                grad_x_t_from_f = grad_z_f @ W[:self.hidden_size, :]
                grad_x_t_from_i = grad_z_i @ W[self.hidden_size:2 * self.hidden_size, :]
                grad_x_t_from_n = grad_z_n @ W[2 * self.hidden_size:, :]
                grad_x_t = grad_x_t_from_f + grad_x_t_from_i + grad_x_t_from_n
                grad_input_layer[:, t, :] = grad_x_t
                grad_h_next = grad_h_prev
            if layer_idx > 0:
                grad_x = grad_input_layer
        return grad_input_layer if self.num_layers > 0 else cp.zeros_like(self.inputs)

class GRU(RecurrentBase):
    def __init__(self, input_size, hidden_size, activation_func=Tanh, num_layers=1, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', decay_rate=0.9, epsilon=1e-8, beta2=0.999):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_func = activation_func
        self.num_layers = num_layers
        super().__init__(weights_shape=(3 * hidden_size, input_size + hidden_size), biases_shape=(1, 3 * hidden_size), decay_rate=decay_rate, epsilon=epsilon, beta2=beta2)
        self.weights = []
        self.biases = []
        self.m_weights = []
        self.v_weights = []
        self.m_biases = []
        self.v_biases = []
        self.squared_grad_weights = []
        self.squared_grad_biases = []
        self.grad_weights = []
        self.grad_biases = []
        current_input_size = input_size
        for layer_idx in range(num_layers):
            if kernel_initializer == 'glorot_uniform':
                limit_ih = cp.sqrt(6 / (current_input_size + 3 * hidden_size))
                w_ih = cp.random.uniform(-limit_ih, limit_ih, (3 * hidden_size, current_input_size))
            else:
                raise ValueError(f"Unsupported kernel_initializer: {kernel_initializer}")
            if recurrent_initializer == 'orthogonal':
                w_hh = orthogonal_init((3 * hidden_size, hidden_size))
            else:
                raise ValueError(f"Unsupported recurrent_initializer: {recurrent_initializer}")
            W = cp.concatenate((w_ih, w_hh), axis=1)
            if bias_initializer == 'zeros':
                b = cp.zeros((1, 3 * hidden_size))
            else:
                raise ValueError(f"Unsupported bias_initializer: {bias_initializer}")
            self._init_layer_params(W, b)
            current_input_size = hidden_size
        self.inputs_per_layer = []
        self.hidden_states_per_layer = []
        self.gates_per_layer = []

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, seq_len, input_size = inputs.shape
        assert input_size == self.input_size, f"Несоответствие размера входа: ожидалось {self.input_size}, получено {input_size}"
        self.inputs_per_layer = []
        self.hidden_states_per_layer = []
        self.gates_per_layer = []
        x = inputs
        for layer_idx in range(self.num_layers):
            self.inputs_per_layer.append(cp.copy(x))
            hidden_states = cp.zeros((batch_size, seq_len + 1, self.hidden_size))
            gates = cp.zeros((batch_size, seq_len, 3 * self.hidden_size))  # z, r, n
            h_prev = cp.zeros((batch_size, self.hidden_size))
            hidden_states[:, 0, :] = h_prev
            W = self.weights[layer_idx]
            b = self.biases[layer_idx]
            for t in range(seq_len):
                combined = cp.concatenate((x[:, t, :], h_prev), axis=1)
                z_full = combined @ W.T + b
                gates[:, t, :] = z_full
                z = Sigmoid()(z_full[:, :self.hidden_size])
                r = Sigmoid()(z_full[:, self.hidden_size:2*self.hidden_size])
                combined_n = cp.concatenate((x[:, t, :], r * h_prev), axis=1)
                z_n = combined_n @ W[2*self.hidden_size:, :].T + b[:, 2*self.hidden_size:]
                n = self.activation_func(z_n)
                h = (1 - z) * n + z * h_prev
                hidden_states[:, t + 1, :] = h
                h_prev = h
            x = hidden_states[:, 1:, :]
            self.hidden_states_per_layer.append(hidden_states)
            self.gates_per_layer.append(gates)
        self.last_hidden = hidden_states[:, -1, :]
        return x

    def backward(self, grad_output):
        batch_size, seq_len, hidden_size = grad_output.shape
        assert hidden_size == self.hidden_size, f"Несоответствие размера скрытого слоя: ожидалось {self.hidden_size}, получено {hidden_size}"
        grad_x = grad_output
        for layer_idx in reversed(range(self.num_layers)):
            x = self.inputs_per_layer[layer_idx]
            hidden_states = self.hidden_states_per_layer[layer_idx]
            gates = self.gates_per_layer[layer_idx]
            W = self.weights[layer_idx]
            grad_input_layer = cp.zeros_like(x)
            grad_h_next = cp.zeros((batch_size, self.hidden_size))
            self.grad_weights[layer_idx][...] = 0
            self.grad_biases[layer_idx][...] = 0
            for t in reversed(range(seq_len)):
                grad_h = grad_x[:, t, :] + grad_h_next
                z = Sigmoid()(gates[:, t, :self.hidden_size])
                r = Sigmoid()(gates[:, t, self.hidden_size:2*self.hidden_size])
                combined_n = cp.concatenate((x[:, t, :], r * hidden_states[:, t, :]), axis=1)
                z_n = combined_n @ W[2*self.hidden_size:, :].T + self.biases[layer_idx][:, 2*self.hidden_size:]
                n = self.activation_func(z_n)
                grad_z = grad_h * (hidden_states[:, t, :] - n)
                grad_n = grad_h * (1 - z)
                grad_h_prev = grad_h * z
                grad_z_n = grad_n * self.activation_func.derivative(z_n)
                grad_combined_n = grad_z_n @ W[2*self.hidden_size:, :]
                grad_x_t_from_n = grad_combined_n[:, :x.shape[2]]
                grad_r_h_prev = grad_combined_n[:, x.shape[2]:]
                grad_r = grad_r_h_prev * hidden_states[:, t, :]
                grad_h_prev_from_r = grad_r_h_prev * r
                grad_z_full_z = grad_z * Sigmoid.derivative(gates[:, t, :self.hidden_size])
                grad_z_full_r = grad_r * Sigmoid.derivative(gates[:, t, self.hidden_size:2*self.hidden_size])
                grad_z_full = cp.concatenate((grad_z_full_z, grad_z_full_r, cp.zeros_like(grad_z_full_z)), axis=1)
                combined = cp.concatenate((x[:, t, :], hidden_states[:, t, :]), axis=1)
                self.grad_weights[layer_idx] += grad_z_full.T @ combined
                self.grad_biases[layer_idx] += cp.sum(grad_z_full, axis=0, keepdims=True)
                self.grad_weights[layer_idx][2*self.hidden_size:, :] += grad_z_n.T @ combined_n
                self.grad_biases[layer_idx][:, 2*self.hidden_size:] += cp.sum(grad_z_n, axis=0, keepdims=True)
                grad_combined_from_gates = grad_z_full[:, :2*self.hidden_size] @ W[:2*self.hidden_size, :]
                grad_x_t_from_gates = grad_combined_from_gates[:, :x.shape[2]]
                grad_h_prev_from_gates = grad_combined_from_gates[:, x.shape[2]:]
                grad_x_t = grad_x_t_from_gates + grad_x_t_from_n
                grad_h_prev += grad_h_prev_from_gates + grad_h_prev_from_r
                grad_input_layer[:, t, :] = grad_x_t
                grad_h_next = grad_h_prev
            if layer_idx > 0:
                grad_x = grad_input_layer
        return grad_input_layer if self.num_layers > 0 else cp.zeros_like(self.inputs)

class minGRU(RecurrentBase):
    def __init__(self, input_size, hidden_size, num_layers=1, kernel_initializer='glorot_uniform', bias_initializer='zeros', decay_rate=0.9, epsilon=1e-8, beta2=0.999):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        super().__init__(weights_shape=(2 * hidden_size, input_size), biases_shape=(1, 2 * hidden_size), decay_rate=decay_rate, epsilon=epsilon, beta2=beta2)
        self.weights = []
        self.biases = []
        self.m_weights = []
        self.v_weights = []
        self.m_biases = []
        self.v_biases = []
        self.squared_grad_weights = []
        self.squared_grad_biases = []
        self.grad_weights = []
        self.grad_biases = []
        current_input_size = input_size
        for layer_idx in range(num_layers):
            if kernel_initializer == 'glorot_uniform':
                limit_ih = cp.sqrt(6 / (current_input_size + hidden_size))
                w_ih = cp.random.uniform(-limit_ih, limit_ih, (2 * hidden_size, current_input_size))
            else:
                raise ValueError(f"Unsupported kernel_initializer: {kernel_initializer}")
            W = w_ih  # No recurrent part
            if bias_initializer == 'zeros':
                b = cp.zeros((1, 2 * hidden_size))
            else:
                raise ValueError(f"Unsupported bias_initializer: {bias_initializer}")
            self._init_layer_params(W, b)
            current_input_size = hidden_size
        self.inputs_per_layer = []
        self.hidden_states_per_layer = []
        self.gates_per_layer = []

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, seq_len, input_size = inputs.shape
        assert input_size == self.input_size, f"Input size mismatch: expected {self.input_size}, got {input_size}"
        self.inputs_per_layer = []
        self.hidden_states_per_layer = []
        self.gates_per_layer = []
        x = inputs
        for layer_idx in range(self.num_layers):
            self.inputs_per_layer.append(cp.copy(x))
            hidden_states = cp.zeros((batch_size, seq_len + 1, self.hidden_size))
            gates = cp.zeros((batch_size, seq_len, 2 * self.hidden_size))  # z, n
            h_prev = cp.zeros((batch_size, self.hidden_size))
            hidden_states[:, 0, :] = h_prev
            W = self.weights[layer_idx]
            b = self.biases[layer_idx]
            for t in range(seq_len):
                x_t = x[:, t, :]
                linear = x_t @ W.T + b
                gates[:, t, :] = linear
                z = Sigmoid()(linear[:, :self.hidden_size])
                n = linear[:, self.hidden_size:]
                h = (1 - z) * n + z * h_prev
                hidden_states[:, t + 1, :] = h
                h_prev = h
            x = hidden_states[:, 1:, :]
            self.hidden_states_per_layer.append(hidden_states)
            self.gates_per_layer.append(gates)
        self.last_hidden = hidden_states[:, -1, :]
        return x

    def backward(self, grad_output):
        batch_size, seq_len, hidden_size = grad_output.shape
        assert hidden_size == self.hidden_size, f"Hidden size mismatch: expected {self.hidden_size}, got {hidden_size}"
        grad_x = grad_output
        for layer_idx in reversed(range(self.num_layers)):
            x = self.inputs_per_layer[layer_idx]
            hidden_states = self.hidden_states_per_layer[layer_idx]
            gates = self.gates_per_layer[layer_idx]
            W = self.weights[layer_idx]
            grad_input_layer = cp.zeros_like(x)
            grad_h_next = cp.zeros((batch_size, self.hidden_size))
            self.grad_weights[layer_idx][...] = 0
            self.grad_biases[layer_idx][...] = 0
            for t in reversed(range(seq_len)):
                grad_h = grad_x[:, t, :] + grad_h_next
                z = Sigmoid()(gates[:, t, :self.hidden_size])
                n = gates[:, t, self.hidden_size:]
                h_prev = hidden_states[:, t, :]
                grad_z = grad_h * (h_prev - n)
                grad_n = grad_h * (1 - z)
                grad_h_prev = grad_h * z
                grad_z_full = grad_z * Sigmoid.derivative(gates[:, t, :self.hidden_size])
                grad_z_n = grad_n  # Derivative of identity is 1
                x_t = x[:, t, :]
                self.grad_weights[layer_idx][:self.hidden_size, :] += grad_z_full.T @ x_t
                self.grad_weights[layer_idx][self.hidden_size:, :] += grad_z_n.T @ x_t
                self.grad_biases[layer_idx][:, :self.hidden_size] += cp.sum(grad_z_full, axis=0, keepdims=True)
                self.grad_biases[layer_idx][:, self.hidden_size:] += cp.sum(grad_z_n, axis=0, keepdims=True)
                grad_x_t_from_z = grad_z_full @ W[:self.hidden_size, :]
                grad_x_t_from_n = grad_z_n @ W[self.hidden_size:, :]
                grad_x_t = grad_x_t_from_z + grad_x_t_from_n
                grad_input_layer[:, t, :] = grad_x_t
                grad_h_next = grad_h_prev
            if layer_idx > 0:
                grad_x = grad_input_layer
        return grad_input_layer if self.num_layers > 0 else cp.zeros_like(self.inputs)

class LastState:
    def forward(self, x):
        self.input_shape = x.shape
        return x[:, -1, :]

    def backward(self, grad_output):
        batch_size, features = grad_output.shape
        seq_len = self.input_shape[1]
        grad_input = cp.zeros(self.input_shape)
        grad_input[:, -1, :] = grad_output
        return grad_input