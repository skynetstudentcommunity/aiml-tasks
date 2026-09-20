# Week 10: PyTorch or TensorFlow — Tutorial

## Why this week matters
Frameworks handle autograd (automatic gradient computation) so you stop
deriving backprop by hand and start iterating fast. Pick one and commit —
PyTorch is used below, but the concepts map directly to TensorFlow/Keras.

---

## 1. Tensors and Autograd

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()          # computes dy/dx automatically
x.grad                # tensor([4.]) -> the gradient at x=2
```

This is the "magic" that replaces the manual backprop math from last week —
PyTorch tracks every operation and can differentiate through it automatically.

---

## 2. Building a Model with nn.Module

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)   # raw logits; loss function applies softmax

model = Net()
```

---

## 3. Training Loop, Batching, DataLoaders

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()
```

Batching (rather than using all data at once) makes training memory-feasible
and, empirically, often helps models converge better.

---

## 4. Regularization

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)
```

- **Dropout**: randomly zeroes some neurons during training, forcing the
  network not to over-rely on any one path — reduces overfitting
- **Batch norm**: normalizes activations between layers, often speeds up and
  stabilizes training
- **Early stopping**: stop training when validation loss stops improving,
  even if training loss keeps dropping (a sign of overfitting)

---

## Project
Rebuild the MNIST classifier in PyTorch (or TensorFlow):
- [ ] Build the model, training loop, and DataLoader
- [ ] Add dropout and/or batch norm
- [ ] Beat your from-scratch NumPy version's accuracy from Week 9

## Resources
- PyTorch official 60-min blitz — https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
- PyTorch Learn the Basics — https://pytorch.org/tutorials/beginner/basics/intro.html
- fast.ai Practical Deep Learning course — https://course.fast.ai/
