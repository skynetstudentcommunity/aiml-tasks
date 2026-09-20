# Week 4: Core ML Concepts — Tutorial

## Why this week matters
Before adding more algorithms, you need the shared vocabulary — otherwise
tutorials, papers, and job interviews will keep tripping you up on terms
everyone assumes you already know.

---

## 1. The Three Flavors of ML

- **Supervised learning**: you have labeled data (input → known output).
  Regression predicts a number; classification predicts a category.
- **Unsupervised learning**: no labels — you're finding structure (clusters,
  patterns) in the data itself.
- **Reinforcement learning**: an agent learns by taking actions and getting
  rewards/penalties (games, robotics) — out of scope for the next 12 weeks,
  but good to know it exists.

---

## 2. Overfitting vs. Underfitting

- **Underfitting**: model is too simple, performs poorly even on training
  data (high bias).
- **Overfitting**: model memorizes training data, performs great there but
  poorly on new data (high variance).

```python
# Symptom you'll actually see:
# train_accuracy = 0.99, val_accuracy = 0.65  -> overfitting
# train_accuracy = 0.60, val_accuracy = 0.58  -> underfitting
```

The **bias-variance tradeoff** is the balancing act: simpler models have
more bias (may miss real patterns) but less variance (more stable);
complex models have less bias but more variance (sensitive to noise in
training data). The right model sits in the middle.

---

## 3. Evaluation Metrics

For **classification**:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Accuracy: % correct overall — misleading on imbalanced data
# Precision: of predicted positives, how many were actually positive?
# Recall: of actual positives, how many did you catch?
# F1: harmonic mean of precision and recall — good single number for
#     imbalanced problems
```

For **regression**:
```python
from sklearn.metrics import mean_squared_error, r2_score

# RMSE: average prediction error, in the same units as your target
# R²: fraction of variance explained by the model (1.0 = perfect)
```

**When to use what:** accuracy is fine for balanced classes; use
precision/recall/F1 when classes are imbalanced (e.g., fraud detection,
where "not fraud" vastly outnumbers "fraud").

---

## 4. Cross-Validation

A single train/val split can be lucky or unlucky. K-fold cross-validation
splits data into K parts, trains K times (each part gets a turn as
validation), and averages the results — a more reliable performance estimate.

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
scores.mean(), scores.std()
```

---

## Project
Write a one-page cheat sheet, in your own words, covering:
- [ ] Supervised vs. unsupervised vs. reinforcement (with an example each)
- [ ] Overfitting vs. underfitting (with a symptom you'd look for)
- [ ] Each metric above, with a one-line "use this when..."
- [ ] Why cross-validation beats a single split

## Resources
- StatQuest: Machine Learning Fundamentals playlist — https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRIHI2ZabQCUmbwql
- Google's Machine Learning Crash Course — https://developers.google.com/machine-learning/crash-course
- Scikit-learn: Cross-validation guide — https://scikit-learn.org/stable/modules/cross_validation.html
