import cupy as cp

class NeuralNetwork:
    def __init__(self, learning_rate=0.001, optimizer="Adam"):
        self.layers = []
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.intermediate_outputs = []

    def add_layer(self, layer):
        if isinstance(layer, list):
            self.layers.extend(layer)
        else:
            self.layers.append(layer)

    def forward(self, X, training=True):
        self.intermediate_outputs = []
        output = X
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'forward') and 'training' in layer.forward.__code__.co_varnames:
                output = layer.forward(output, training=training)
            else:
                output = layer.forward(output)
            self.intermediate_outputs.append(output)
        return output

    def backward(self, y, loss_func):
        output = self.intermediate_outputs[-1]
        loss_gradient = loss_func.gradient(y, output)
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                loss_gradient = layer.backward(loss_gradient)
            else:
                print(f"Предупреждение: у слоя {layer} нет метода backward.")

    def update_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'update_weights'):
                layer.update_weights(self.learning_rate, optimizer=self.optimizer)