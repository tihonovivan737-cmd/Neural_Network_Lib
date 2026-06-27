# Neural Network Lib

GPU-accelerated deep learning library on top of `CuPy`. The project includes basic building blocks for fully connected, convolutional, and recurrent neural networks, plus a simple BPE tokenizer and model serialization to `HDF5`.

## Features

- Dense neural networks with configurable activations
- Convolutional layers: `Conv2D`, `BatchNorm2D`, `MaxPooling2D`, `AveragePooling2D`
- Recurrent layers: `RNN`, `LSTM`, `GRU`, `minLSTM`, `minGRU`
- Utility layers: `Flatten`, `Reshape`, `Embedding`, `LastState`
- Activations: `ReLU`, `LeakyReLU`, `SReLU`, `SiLU`, `Sigmoid`, `Tanh`, `Softmax`
- Loss functions: `MeanSquaredError`, `CrossEntropyLoss`, `HingeLoss`
- Optimizers supported by layers: `SGD`, `AdaGrad`, `RMSProp`, `Adam`, `AdaDelta`
- BPE tokenizer training, saving, loading, encoding, and decoding
- Model save/load helpers in `HDF5`

## Project Structure

```text
.
|-- activations.py
|-- CNN.py
|-- layers.py
|-- losses.py
|-- network.py
|-- network_save_and_load.py
|-- RNN.py
|-- tokenizer.py
|-- train.py
|-- __init__.py
```

## Requirements

- Python 3.10+
- NVIDIA GPU and CUDA-compatible `CuPy` build for GPU execution
- Python packages from `requirements.txt`

Install base dependencies:

```bash
pip install -r requirements.txt
```

Then install the correct `CuPy` package for your CUDA version, for example:

```bash
pip install cupy-cuda12x
```

If you use another CUDA version, replace the package accordingly.

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/Neural_Network_Lib.git
cd Neural_Network_Lib
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install cupy-cuda12x
```

## Quick Start

Example of a simple fully connected classifier:

```python
import cupy as cp

from network import NeuralNetwork
from layers import DenseLayer
from activations import ReLU, Softmax
from losses import CrossEntropyLoss
from train import train

x_train = cp.random.randn(128, 32)
y_indices = cp.random.randint(0, 3, size=128)
y_train = cp.eye(3)[y_indices]

model = NeuralNetwork(learning_rate=0.001, optimizer="Adam")
model.add_layer(DenseLayer(32, 64, ReLU()))
model.add_layer(DenseLayer(64, 3, Softmax()))

loss = CrossEntropyLoss()
accuracy, epoch_loss = train(model, x_train, y_train, loss, batch_size=32)

print("accuracy:", accuracy)
print("loss:", epoch_loss)
```

## Data Format Expectations

- Dense layers expect tensors shaped like `(batch_size, features)`
- Convolution layers expect `(batch_size, height, width, channels)`
- Recurrent layers expect `(batch_size, sequence_length, features)`
- Classification training helpers assume one-hot encoded targets

## Model Saving

```python
from network_save_and_load import save_model, load_model

save_model(model, "model.h5")
restored_model = load_model("model.h5")
```

## Tokenizer Example

```python
from tokenizer import train_bpe, load_tokenizer, encode, decode

_, merge_rules, _ = train_bpe(
    ["Neural networks are fun", "Tokenizers are useful"],
    vocab_size=100,
    file_path="tokenizer.json"
)

merge_rules, encoder, decoder = load_tokenizer("tokenizer.json")
token_ids = encode("Neural networks", merge_rules, encoder)
text = decode(token_ids, decoder)
```

## Notes

- The library is focused on educational and experimental use.
- The codebase currently provides core modules, but does not yet include a packaged release, benchmark suite, or automated tests.
- `CuPy` installation depends on your local CUDA setup, so it is intentionally documented separately from the base requirements.

## License

Add your preferred license before publishing the project publicly.
