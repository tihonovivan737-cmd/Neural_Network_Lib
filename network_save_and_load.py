import h5py
import cupy as cp
from __init__ import MaxPooling2D, Flatten, BatchNorm2D, DenseLayer, Conv2D, NeuralNetwork, ReLU, Sigmoid, \
    Softmax, Tanh, AveragePooling2D


def save_model(network, file_path):
    with h5py.File(file_path, 'w') as f:
        f.attrs['learning_rate'] = network.learning_rate
        f.attrs['optimizer'] = network.optimizer
        for i, layer in enumerate(network.layers):
            group = f.create_group(f'layer_{i}')
            group.attrs['type'] = layer.__class__.__name__
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                group.create_dataset('weights', data=cp.asnumpy(layer.weights))
                group.create_dataset('biases', data=cp.asnumpy(layer.biases))
                if isinstance(layer, DenseLayer):
                    group.attrs['activation'] = layer.activation_func.__class__.__name__
                elif isinstance(layer, Conv2D):
                    group.attrs['activation'] = layer.activation_func.__class__.__name__ if layer.activation_func else 'None'
                    group.attrs['num_filters'] = layer.params['num_filters']
                    group.attrs['kernel_size'] = layer.params['kernel_size']
                    group.attrs['stride'] = layer.params['stride']
                    group.attrs['padding'] = layer.params['padding']
                    group.attrs['in_channels'] = layer.params['in_channels']
                elif isinstance(layer, BatchNorm2D):
                    group.create_dataset('running_mean', data=cp.asnumpy(layer.running_mean))
                    group.create_dataset('running_var', data=cp.asnumpy(layer.running_var))
                    group.attrs['num_features'] = layer.num_features
                    group.attrs['momentum'] = layer.momentum
                    group.attrs['epsilon'] = layer.epsilon
            elif isinstance(layer, MaxPooling2D):
                group.attrs['pool_size'] = layer.pool_size
                group.attrs['stride'] = layer.stride

def load_model(file_path):
    with h5py.File(file_path, 'r') as f:
        learning_rate = f.attrs['learning_rate']
        optimizer = f.attrs['optimizer']
        network = NeuralNetwork(learning_rate=learning_rate, optimizer=optimizer)
        activation_mapping = {
            'ReLU': ReLU,
            'Sigmoid': Sigmoid,
            'Softmax': Softmax,
            'Tanh': Tanh,
            'None': None
        }
        for i in range(len(f.keys())):
            group = f[f'layer_{i}']
            layer_type = group.attrs['type']
            if layer_type == 'DenseLayer':
                weights = cp.asarray(group['weights'])
                biases = cp.asarray(group['biases'])
                activation_name = group.attrs['activation']
                activation_func = activation_mapping[activation_name]()
                layer = DenseLayer(weights.shape[0], weights.shape[1], activation_func)
                layer.weights = weights
                layer.biases = biases
            elif layer_type == 'Conv2D':
                weights = cp.asarray(group['weights'])
                biases = cp.asarray(group['biases'])
                activation_name = group.attrs['activation']
                activation_func = activation_mapping[activation_name]()
                layer = Conv2D(
                    num_filters=group.attrs['num_filters'],
                    kernel_size=group.attrs['kernel_size'],
                    stride=group.attrs['stride'],
                    padding=group.attrs['padding'],
                    in_channels=group.attrs['in_channels'],
                    weights_initializer='he',
                    activation_func=activation_func
                )
                layer.weights = weights
                layer.biases = biases
            elif layer_type == 'BatchNorm2D':
                weights = cp.asarray(group['weights'])
                biases = cp.asarray(group['biases'])
                running_mean = cp.asarray(group['running_mean'])
                running_var = cp.asarray(group['running_var'])
                layer = BatchNorm2D(
                    num_features=group.attrs['num_features'],
                    momentum=group.attrs['momentum'],
                    eps=group.attrs['epsilon']
                )
                layer.weights = weights
                layer.biases = biases
                layer.running_mean = running_mean
                layer.running_var = running_var
            elif layer_type == 'MaxPooling2D':
                layer = MaxPooling2D(
                    pool_size=group.attrs['pool_size'],
                    stride=group.attrs['stride']
                )
            elif layer_type == 'A':
                layer = MaxPooling2D(
                    pool_size=group.attrs['pool_size'],
                    stride=group.attrs['stride']
                )
            elif layer_type == 'Flatten':
                layer = Flatten()
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            network.add_layer(layer)
    return network