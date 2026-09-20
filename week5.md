# Week 5: Regression & Classification with Scikit-learn — Tutorial

## Why this week matters
Now that you understand the theory, you'll use sklearn's consistent API
(`fit`, `predict`) to train real models fast, and compare a few classic
algorithms head-to-head.

---

## 1. Linear & Logistic Regression

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

# Regression: predicting a continuous number
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
preds = lin_model.predict(X_test)

# Classification: predicting a category (despite the name "regression")
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
preds = log_model.predict(X_test)
probs = log_model.predict_proba(X_test)   # confidence per class
```

Logistic regression works by squashing a linear combination of features
through a sigmoid function into a 0–1 probability.

---

## 2. K-Nearest Neighbors (KNN)

Classifies a new point by looking at the K closest points in the training
data and taking a majority vote. No real "training" happens — all the work
is at prediction time.

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

KNN is sensitive to feature scale (revisit Week 3's scaling step) and slows
down on large datasets since it compares against every training point.

---

## 3. Decision Trees

Splits data repeatedly on feature thresholds ("is age > 30?") to reach a
prediction — easy to visualize and explain.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X_train, y_train)

import matplotlib.pyplot as plt
plot_tree(tree, feature_names=X_train.columns, filled=True)
plt.show()
```

`max_depth` controls overfitting — an unrestricted tree will memorize the
training data.

---

## 4. Confusion Matrix & ROC Curves

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

cm = confusion_matrix(y_test, preds)
ConfusionMatrixDisplay(cm).plot()

fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1])
auc = roc_auc_score(y_test, probs[:, 1])
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
```

The confusion matrix shows exactly which classes get mixed up (not just an
overall score). ROC/AUC shows how well the model separates classes across
all possible decision thresholds — useful when you might adjust the
threshold later (e.g., trading precision for recall).

---

## Project
Build a classifier (e.g., loan default / disease diagnosis):
- [ ] Train Logistic Regression, KNN, and a Decision Tree on the same split
- [ ] Compare accuracy, precision, recall, F1 in a table
- [ ] Plot confusion matrix + ROC curve for your best model

## Resources
- Scikit-learn: Supervised learning docs — https://scikit-learn.org/stable/supervised_learning.html
- StatQuest: Logistic Regression — https://www.youtube.com/watch?v=yIYKR4sgzI8
- StatQuest: Decision Trees — https://www.youtube.com/watch?v=7VeUPuFGJHk
