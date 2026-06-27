# Neural Network Lib

Библиотека глубокого обучения на базе `CuPy` с GPU-ускорением. Проект включает базовые компоненты для полносвязных, сверточных и рекуррентных нейронных сетей, а также простой BPE-токенизатор и сохранение моделей в формате `HDF5`.

## Возможности

- Полносвязные нейронные сети с настраиваемыми функциями активации
- Сверточные слои: `Conv2D`, `BatchNorm2D`, `MaxPooling2D`, `AveragePooling2D`
- Рекуррентные слои: `RNN`, `LSTM`, `GRU`, `minLSTM`, `minGRU`
- Вспомогательные слои: `Flatten`, `Reshape`, `Embedding`, `LastState`
- Функции активации: `ReLU`, `LeakyReLU`, `SReLU`, `SiLU`, `Sigmoid`, `Tanh`, `Softmax`
- Функции потерь: `MeanSquaredError`, `CrossEntropyLoss`, `HingeLoss`
- Поддерживаемые оптимизаторы: `SGD`, `AdaGrad`, `RMSProp`, `Adam`, `AdaDelta`
- Обучение BPE-токенизатора, сохранение, загрузка, кодирование и декодирование
- Сохранение и загрузка моделей через `HDF5`

## Структура Проекта

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

## Требования

- Python 3.10+
- NVIDIA GPU и совместимая с вашей CUDA сборка `CuPy`
- Python-зависимости из `requirements.txt`

Установка базовых зависимостей:

```bash
pip install -r requirements.txt
```

Дальше нужно установить подходящий пакет `CuPy` под вашу версию CUDA, например:

```bash
pip install cupy-cuda12x
```

Если у вас другая версия CUDA, замените пакет на соответствующий.

## Установка

Клонируйте репозиторий:

```bash
git clone https://github.com/tihonovivan737-cmd/Neural_Network_Lib.git
cd Neural_Network_Lib
```

Установите зависимости:

```bash
pip install -r requirements.txt
pip install cupy-cuda12x
```

## Быстрый Старт

Пример простого полносвязного классификатора:

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

## Формат Данных

- Полносвязные слои ожидают тензоры формы `(batch_size, features)`
- Сверточные слои ожидают `(batch_size, height, width, channels)`
- Рекуррентные слои ожидают `(batch_size, sequence_length, features)`
- Вспомогательные функции обучения для классификации предполагают one-hot разметку таргетов

## Сохранение Модели

```python
from network_save_and_load import save_model, load_model

save_model(model, "model.h5")
restored_model = load_model("model.h5")
```

## Пример С Токенизатором

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

## Примечания

- Библиотека больше ориентирована на обучение, эксперименты и личные проекты.
- В проекте есть основные модули, но пока нет оформленного пакетного релиза, набора бенчмарков и автоматических тестов.
- `CuPy` ставится отдельно, потому что выбор пакета зависит от вашей локальной версии CUDA.
