# Week 11: CNNs (Computer Vision) — Tutorial

## Why this week matters
Fully-connected networks don't scale to images (too many parameters, no
sense of spatial structure). CNNs exploit the fact that nearby pixels are
related, using shared filters that slide across the image.

---

## 1. Convolutions and Pooling

A **convolution** slides a small filter (e.g., 3x3) across the image,
computing a weighted sum at each position — early filters learn to detect
edges, later ones detect more complex shapes.

```python
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
pool = nn.MaxPool2d(kernel_size=2)   # halves spatial size, keeps strongest signal
```

**Pooling** downsamples the feature maps, reducing computation and adding a
bit of translation invariance (small shifts in the image matter less).

---

## 2. A Simple CNN Architecture

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)  # depends on input size

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(x)
```

This is roughly the shape of classic architectures like LeNet — stack
conv → activation → pool blocks, then flatten into fully-connected layers.

---

## 3. Data Augmentation

Artificially expands your training set by applying random transformations,
reducing overfitting on small image datasets.

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
])
```

---

## 4. Transfer Learning

Instead of training from scratch, start from a model pretrained on millions
of images (ImageNet) and fine-tune it on your smaller dataset — usually far
better results with far less data.

```python
from torchvision import models

model = models.resnet18(weights="IMAGENET1K_V1")
for param in model.parameters():
    param.requires_grad = False          # freeze pretrained layers

model.fc = nn.Linear(model.fc.in_features, num_classes)  # replace final layer
# Now only the new final layer trains (fast); optionally unfreeze more later
```

---

## Project
Build an image classifier using transfer learning:
- [ ] Pick a dataset (own photos or a Kaggle image dataset)
- [ ] Apply data augmentation to the training set
- [ ] Fine-tune a pretrained ResNet18
- [ ] Report accuracy and show a few example predictions

## Resources
- CS231n: Convolutional Neural Networks — https://cs231n.github.io/convolutional-networks/
- PyTorch: Transfer Learning tutorial — https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- Papers with Code: Image Classification — https://paperswithcode.com/task/image-classification
