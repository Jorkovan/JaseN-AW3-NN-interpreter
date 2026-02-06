# AW3 NN JaseN Interpreter
### [!CAUTION!] This project was developed and tested using PyCharm. Exercise caution when using it outside of controlled environments or with non-standard Python distributions.
This is a ***lightweight, pure-Python and NumPy implementation*** of a feedforward neural network designed specifically for the ***JaseN NN Protocol***. This project allows you to build, train, and export neural networks directly into ***Ancient Warfare 3...***
### 1. Core Features
***JaseN Protocol Support:*** Seamlessly export trained weights/biases for use in AW3.

***Multiple Activation Functions:*** Includes ReLU, Sigmoid, Tanh, Leaky ReLU, and Softmax.

***Modular Architecture:*** Define input, hidden, and output layers with a simple configuration list.

***Backpropagation:*** Built-in gradient descent for model optimization.

##  Installation:
1. Clone this repository to your PyCharm env.

2. Navigate to the directory and install dependencies:
```bash
cd NeuralNetInterpreter
pip install -r requirements.txt
```

## 2. How to Use:

### Define and Create a Network:
The network is defined as a list of layers. Format: [LayerType, Size, ActivationFunction]

***[!IMPORTANT!] A valid network must contain exactly one Input (i) layer and one Output (o) layer.***

LayerType: "i" (Input), "h" (Hidden), or "o" (Output).

Activation: "ReLu", "Sigmoid", "Tanh", "LeakyReLu", or "Softmax".

Define a network: 3 Inputs -> 5 Hidden (ReLU) -> 2 Outputs (Sigmoid)
Propagating the network requires an INPUT layer, make SURE there are as many INPUTS as weights in the first HIDDEN layer.
```python
import NeuralNetInterpreter.Net as sn
# Example: 3 Inputs -> 5 Hidden (ReLU) -> 2 Outputs (Sigmoid)
prefs = [
    ["i", 3, "ReLu"],
    ["h", 5, "ReLu"],
    ["o", 2, "Sigmoid"]
]
my_net = sn.CreateNet(prefs)
```

### Forward Propagation:
Propagating the network requires an INPUT layer, make SURE there are as many INPUTS as weights in the first HIDDEN layer.
Ensure the number of inputs matches the size of your input layer.
```python
inputs = [0.5, -1.2, 3.3]
my_net = sn.EditNetInputs(my_net, inputs)
output = sn.PropagateNet(my_net)

print(f"Network Output: {output}")
```

### Training (Backpropagation):
You can train the network by calculating the error (Difference between desired target and actual output) and applying a learning rate.
```python
# Standard supervised learning error calculation
error = target_values - output
learning_rate = 0.01

my_net = sn.BackPropagate(my_net, error, learning_rate)
```

### Exporting to AW3:
And finaly, to export your Neural Network, you will need a NETSTRING...
To generate the string required for the JaseN Protocol in Ancient Warfare 3:
```python
import NeuralNetInterpreter.Net as sn

exported_data = ExportNet("MyModel", my_net)
print(exported_data)
```

# Technical Math
The backpropagation in this script utilizes the chain rule to update weights based on the slope (derivative) of the activation functions. For example, for the Sigmoid function:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$The gradient used for updates is:$$\sigma'(y) = y \cdot (1 - y)$$