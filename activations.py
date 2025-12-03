import cupy as cp


class LeakyReLU:
    """
    Класс для функции активации Leaky ReLU (Leaky Rectified Linear Unit).
    """

    def __call__(self, z):
        """
        Вычисляет значение функции Leaky ReLU.
        LeakyReLU(z) = max(alpha * z, z)
        """
        return cp.where(z > 0, z, 0.01 * z)

    def derivative(self, z):
        """
        Вычисляет производную функции Leaky ReLU.
        Производная равна 1 для z > 0 и alpha для z <= 0.
        """
        derivative = cp.ones_like(z)
        derivative[z <= 0] = 0.01
        return derivative


class SReLU:
    """
    Класс для функции активации SReLU (S-shaped Rectified Linear Unit).
    """

    def __init__(self, tl=-1, al=0.01, tr=1, ar=0.01):
        """
        Инициализация параметров SReLU.
        tl, tr - пороговые значения,
        al, ar - коэффициенты наклона.
        """
        self.tl = tl
        self.al = al
        self.tr = tr
        self.ar = ar

    def __call__(self, z):
        """
        Вычисляет значение функции SReLU.
        """
        return cp.where(z < self.tl, self.tl + self.al * (z - self.tl),
                        cp.where(z > self.tr, self.tr + self.ar * (z - self.tr), z))

    def derivative(self, z):
        """
        Вычисляет производную функции SReLU.
        """
        return cp.where(z < self.tl, self.al, cp.where(z > self.tr, self.ar, 1))


class SiLU:
    def __call__(self, z):
        self.z = z
        sigmoid = 1 / (1 + cp.exp(-z))
        return z * sigmoid

    @staticmethod
    def derivative(z):
        sigmoid = 1 / (1 + cp.exp(-z))
        return sigmoid + z * sigmoid * (1 - sigmoid)


class ReLU:
    """
        Класс для функции активации ReLU (Rectified Linear Unit).
        """

    def __call__(self, z):
        """
        Вычисляет значение функции ReLU для входного массива z.
        ReLU(z) = max(0, z)

        Параметры:
        z (ndarray): Входной массив.

        Возвращает:
        ndarray: Массив после применения функции ReLU.
        """
        return cp.maximum(0, z)

    @staticmethod
    def derivative(z):
        """
        Вычисляет производную функции ReLU по входу z.
        Производная равна 1 для z > 0 и 0 для z <= 0.

        Параметры:
        z (ndarray): Входной массив.

        Возвращает:
        ndarray: Массив значений производной функции ReLU.
        """
        derivative = cp.ones_like(z)
        derivative[z <= 0] = 0
        return derivative


class Sigmoid:
    def __call__(self, z):
        """
        Вычисляет значение сигмоидной функции для входного массива z.
        Sigmoid(z) = 1 / (1 + exp(-z))

        Параметры:
        z (ndarray): Входной массив.

        Возвращает:
        ndarray: Массив после применения сигмоиды.
        """
        return 1 / (1 + cp.exp(-z))

    @staticmethod
    def derivative(z):
        """
        Вычисляет производную сигмоидной функции по входу z.
        Производная: sigmoid(z) * (1 - sigmoid(z))

        Параметры:
        z (ndarray): Входной массив.

        Возвращает:
        ndarray: Массив значений производной сигмоиды.
        """
        sigmoid = 1 / (1 + cp.exp(-z))
        return sigmoid * (1 - sigmoid)


class Tanh:
    """
       Класс для функции активации Tanh (гиперболический тангенс).
       """

    def __call__(self, z):
        """
        Вычисляет значение функции гиперболического тангенса для z.
        Tanh(z) = tanh(z)

        Параметры:
        z (ndarray): Входной массив.

        Возвращает:
        ndarray: Массив после применения tanh.
        """
        return cp.tanh(z)

    @staticmethod
    def derivative(z):
        """
        Вычисляет производную функции гиперболического тангенса по входу z.
        Производная: 1 - tanh(z)^2

        Параметры:
        z (ndarray): Входной массив.

        Возвращает:
        ndarray: Массив значений производной функции tanh.
        """
        return 1 - cp.tanh(z) ** 2


class Softmax:
    def __call__(self, z):
        exp_z = cp.exp(z - cp.max(z, axis=1, keepdims=True))
        return exp_z / cp.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def derivative(z):
        raise NotImplementedError("Softmax derivative should not be called directly")


class ActivationLayer:
    def __init__(self, activation_func):
        self.activation_func = activation_func

    def forward(self, inputs):
        self.z = inputs
        return self.activation_func(inputs)

    def backward(self, grad_output):
        """
        Умножаем входящий градиент на производную активации, используя сохраненный z.
        """
        grad_activation = self.activation_func.derivative(self.z)
        return grad_output * grad_activation
