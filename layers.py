import cupy as cp
from activations import Softmax


class LayerWithParameters:
    def __init__(self, weights_shape, biases_shape, decay_rate=1, epsilon=1e-8):
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.weights = None
        self.biases = None
        self.squared_grad_weights = cp.zeros(weights_shape)
        self.squared_grad_biases = cp.zeros(biases_shape)
        self.m_weights = cp.zeros(weights_shape)
        self.v_weights = cp.zeros(weights_shape)
        self.m_biases = cp.zeros(biases_shape)
        self.v_biases = cp.zeros(biases_shape)
        self.grad_weights = None
        self.grad_biases = None
        self.t = 1

    def update_weights(self, learning_rate=0.001, optimizer="Adam", weight_decay=0):
        clip_value = 5
        norm_weights = cp.sqrt(cp.sum(self.grad_weights ** 2))
        norm_biases = cp.sqrt(cp.sum(self.grad_biases ** 2))
        if norm_weights > clip_value:
            self.grad_weights = self.grad_weights * clip_value / norm_weights
        if norm_biases > clip_value:
            self.grad_biases = self.grad_biases * clip_value / norm_biases
        if optimizer == "RMSProp":
            self.squared_grad_weights = self.decay_rate * self.squared_grad_weights + (1 - self.decay_rate) * (
                    self.grad_weights ** 2)
            self.squared_grad_biases = self.decay_rate * self.squared_grad_biases + (1 - self.decay_rate) * (
                    self.grad_biases ** 2)
            self.weights -= learning_rate * self.grad_weights / (cp.sqrt(self.squared_grad_weights) + self.epsilon)
            self.biases -= learning_rate * self.grad_biases / (cp.sqrt(self.squared_grad_biases) + self.epsilon)
        elif optimizer == "AdaGrad":
            self.squared_grad_weights += self.grad_weights ** 2
            self.squared_grad_biases += self.grad_biases ** 2
            self.weights -= learning_rate * self.grad_weights / (cp.sqrt(self.squared_grad_weights) + self.epsilon)
            self.biases -= learning_rate * self.grad_biases / (cp.sqrt(self.squared_grad_biases) + self.epsilon)
        elif optimizer == "Adam":
            self.grad_weights = self.grad_weights + weight_decay * self.weights
            self.grad_biases = self.grad_biases + weight_decay * self.biases
            self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * self.grad_weights
            self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (self.grad_weights ** 2)
            self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * self.grad_biases
            self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (self.grad_biases ** 2)
            m_weights_hat = self.m_weights / (1 - self.beta1 ** self.t)
            v_weights_hat = self.v_weights / (1 - self.beta2 ** self.t)
            m_biases_hat = self.m_biases / (1 - self.beta1 ** self.t)
            v_biases_hat = self.v_biases / (1 - self.beta2 ** self.t)
            self.weights -= learning_rate * m_weights_hat / (cp.sqrt(v_weights_hat) + self.epsilon)
            self.biases -= learning_rate * m_biases_hat / (cp.sqrt(v_biases_hat) + self.epsilon)
            self.t += 1
        elif optimizer == "AdaDelta":
            self.squared_grad_weights = self.decay_rate * self.squared_grad_weights + (1 - self.decay_rate) * (
                    self.grad_weights ** 2)
            self.squared_grad_biases = self.decay_rate * self.squared_grad_biases + (1 - self.decay_rate) * (
                    self.grad_biases ** 2)
            update_weights = -self.grad_weights * (
                    cp.sqrt(self.m_weights + self.epsilon) / cp.sqrt(self.squared_grad_weights + self.epsilon))
            update_biases = -self.grad_biases * (
                    cp.sqrt(self.m_biases + self.epsilon) / cp.sqrt(self.squared_grad_biases + self.epsilon))
            self.m_weights = self.decay_rate * self.m_weights + (1 - self.decay_rate) * (update_weights ** 2)
            self.m_biases = self.decay_rate * self.m_biases + (1 - self.decay_rate) * (update_biases ** 2)
            self.weights += update_weights
            self.biases += update_biases
        elif optimizer == "SGD":
            self.weights -= learning_rate * self.grad_weights
            self.biases -= learning_rate * self.grad_biases
        self.grad_weights = None
        self.grad_biases = None


class DenseLayer(LayerWithParameters):
    def __init__(self, input_size, output_size, activation_func, weights_initializer='he',
                 biases_initializer='uniform', learning_rate=0.001, decay_rate=1, epsilon=1e-8):
        super().__init__(weights_shape=(input_size, output_size), biases_shape=(1, output_size),
                         decay_rate=decay_rate, epsilon=epsilon)
        self.activation_func = activation_func
        self.learning_rate = learning_rate
        if weights_initializer == 'random':
            self.weights = cp.random.randn(input_size, output_size) * 0.01
        elif weights_initializer == 'xavier':
            self.weights = cp.random.randn(input_size, output_size) * cp.sqrt(1 / input_size)
        elif weights_initializer == 'he':
            self.weights = cp.random.randn(input_size, output_size) * cp.sqrt(2 / input_size)
        elif weights_initializer == 'normal':
            self.weights = cp.random.normal(0, 1, (input_size, output_size))
        else:
            raise ValueError(f"Unknown weights initializer: {weights_initializer}")
        if biases_initializer == 'zeros':
            self.biases = cp.zeros((1, output_size))
        elif biases_initializer == 'ones':
            self.biases = cp.ones((1, output_size))
        elif biases_initializer == 'normal':
            self.biases = cp.random.normal(0, 1, (1, output_size))
        elif biases_initializer == 'uniform':
            limit = cp.sqrt(6.0 / input_size)
            self.biases = cp.random.uniform(-limit, limit, (1, output_size))
        else:
            raise ValueError(f"Unknown biases initializer: {biases_initializer}")

    def forward(self, inputs):
        self.inputs = inputs
        self.z = cp.dot(inputs, self.weights) + self.biases
        if self.activation_func is not None:
            self.a = self.activation_func(self.z)
            return self.a
        return self.z

    def backward(self, grad_output):
        if isinstance(self.activation_func, Softmax):
            grad_input = cp.dot(grad_output, self.weights.T)
            self.grad_weights = cp.dot(self.inputs.T, grad_output)
            self.grad_biases = cp.sum(grad_output, axis=0, keepdims=True)
        else:
            grad_activation = 1.0 if self.activation_func is None else self.activation_func.derivative(self.z)
            grad_output = grad_output * grad_activation
            grad_input = cp.dot(grad_output, self.weights.T)
            self.grad_weights = cp.dot(self.inputs.T, grad_output)
            self.grad_biases = cp.sum(grad_output, axis=0, keepdims=True)
        return grad_input

class Flatten:
    def forward(self, inputs):
        self.input_shape = inputs.shape
        batch_size = inputs.shape[0]
        return inputs.reshape(batch_size, -1)

    def backward(self, d_out):
        return d_out.reshape(self.input_shape)


class Reshape:
    def __init__(self, target_shape):
        self.target_shape = target_shape
        self.input_shape = None

    def forward(self, inputs):
        self.input_shape = inputs.shape
        return inputs.reshape(self.target_shape)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)




