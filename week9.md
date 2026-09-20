# Week 9: Neural Network Fundamentals — Tutorial

## Why this week matters
Everything from here on (CNNs, transformers, LLMs) is built on the same
core idea: layers of simple units, connected, trained by backpropagation.
Building one from scratch once demystifies all of it.

---

## 1. The Perceptron and Activation Functions

A neuron computes a weighted sum of inputs, adds a bias, then passes it
through a nonlinear **activation function**. Without nonlinearity, stacking
layers would collapse into one linear function — activations are what let
networks learn complex patterns.

```python
import numpy as np

def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)
```

- **ReLU**: default for hidden layers, fast, avoids some training issues
- **Sigmoid**: squashes to 0–1, good for binary output
- **Softmax**: turns scores into a probability distribution, used for
  multi-class output layers

---

## 2. Forward Propagation

Data flows layer by layer: `output = activation(weights · input + bias)`,
with each layer's output feeding the next.

```python
def forward(x, W1, b1, W2, b2):
    z1 = x @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)
    return a2
```

---

## 3. Backpropagation (Conceptually)

After a forward pass, you get a **loss** (how wrong the prediction was).
Backpropagation uses the chain rule to compute how much each weight
contributed to that error, working backward from the output layer to the
input layer — then gradient descent nudges every weight to reduce the loss.

Do this once by hand on paper for a tiny 2-layer network before coding it —
it's the single best way to make it click.

---

## 4. Loss Functions and Optimizers

```python
# Cross-entropy loss for classification
def cross_entropy(preds, labels):
    return -np.sum(labels * np.log(preds + 1e-9)) / len(labels)
```

- **SGD**: updates weights using the gradient from a small batch at a time
- **Adam**: adapts the learning rate per-parameter, usually converges faster
  and more reliably than plain SGD — a safe default going forward

---

## 5. Building a NN From Scratch (NumPy only)

```python
class SimpleNN:
    def __init__(self, in_size, hidden_size, out_size):
        self.W1 = np.random.randn(in_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, out_size) * 0.01
        self.b2 = np.zeros(out_size)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    # backward() implements the gradients derived by hand above,
    # then updates W1, b1, W2, b2 via gradient descent
```

---

## Project
Train your from-scratch NN on MNIST digits:
- [ ] Implement forward pass, loss, and backward pass by hand
- [ ] Train for several epochs, tracking loss
- [ ] Report final test accuracy

## Resources
- 3Blue1Brown: Neural Networks playlist — https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- Michael Nielsen: Neural Networks and Deep Learning (free book) — http://neuralnetworksanddeeplearning.com/
- MNIST dataset — http://yann.lecun.com/exdb/mnist/
