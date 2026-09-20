# Week 2: Math You Actually Need — Tutorial

## Why this week matters
You don't need a math degree to do ML, but every algorithm you'll touch is
built from three ingredients: linear algebra (how data is represented),
calculus (how models learn), and probability (how uncertainty is handled).
This week builds just enough intuition to stop treating models as black boxes.

---

## 1. Linear Algebra — vectors and matrices

A **vector** is a list of numbers (one data point's features). A **matrix**
is a table of vectors (your whole dataset).

```python
import numpy as np

v = np.array([1, 2, 3])           # a single data point
X = np.array([[1, 2], [3, 4]])    # a dataset: 2 rows, 2 features

# Dot product: multiply matching elements, sum them
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
np.dot(v1, v2)   # 1*4 + 2*5 + 3*6 = 32
```

The dot product is everywhere in ML: a linear model's prediction is just
`dot(weights, features) + bias`. Matrix multiplication is doing many dot
products at once — one per row of data, against one set of weights.

---

## 2. Calculus — derivatives and gradients

A **derivative** tells you the slope of a function at a point — which
direction to move to increase or decrease it. A **gradient** is the same
idea generalized to functions of many variables (like a model's weights).

Intuition: if you're on a hillside in fog and want to reach the bottom, you
feel which way is steepest downhill and take a step that way. That's
gradient descent.

```python
# f(x) = x^2, derivative is f'(x) = 2x
def f(x): return x**2
def grad(x): return 2*x

x = 10.0
lr = 0.1
for i in range(20):
    x = x - lr * grad(x)   # step downhill
print(x)   # converges toward 0, the minimum
```

---

## 3. Probability & Statistics

- **Mean/variance**: center and spread of your data.
- **Distributions**: shapes data tends to follow (normal/Gaussian is the
  most common assumption in ML).
- **Bayes' theorem**: updates a belief given new evidence —
  `P(A|B) = P(B|A) * P(A) / P(B)`. This underlies Naive Bayes classifiers
  and a lot of probabilistic reasoning in ML.

```python
data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
data.mean()   # 5.0
data.var()    # spread around the mean
data.std()    # sqrt(variance), same units as the data
```

---

## 4. Gradient Descent From Scratch

This is the algorithm that trains almost every model you'll build for the
next 15 weeks, so implementing it once by hand matters.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x): return x**2
def grad(x): return 2*x

x = 10.0
lr = 0.1
history = []

for step in range(30):
    x = x - lr * grad(x)
    history.append(f(x))

plt.plot(history)
plt.xlabel("Step")
plt.ylabel("f(x)")
plt.title("Loss curve: gradient descent on f(x) = x^2")
plt.show()
```

Notice the loss drops fast at first, then slows down as you approach the
minimum — this shape (steep, then flattening) is what you'll see in every
training loss curve you plot from now on.

---

## Project
Write from-scratch linear regression (no sklearn) using gradient descent:
- [ ] Generate or load simple (x, y) data
- [ ] Initialize weight and bias, define the MSE loss
- [ ] Loop: predict → compute loss → compute gradients → update weights
- [ ] Plot the loss curve over iterations

## Resources
- 3Blue1Brown: Essence of Linear Algebra — https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
- 3Blue1Brown: Essence of Calculus — https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr
- StatQuest: Statistics Fundamentals — https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9
- Khan Academy: Linear Algebra — https://www.khanacademy.org/math/linear-algebra
