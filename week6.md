# Week 6: Ensemble Methods — Tutorial

## Why this week matters
A single decision tree overfits easily. Ensembles combine many weaker
models into one stronger, more stable predictor — this is what wins most
Kaggle tabular-data competitions.

---

## 1. Random Forests (Bagging)

Trains many decision trees on random subsets of data and features, then
averages their predictions (or majority-votes for classification). Random
subsets make individual trees less correlated, so their errors cancel out.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)
```

---

## 2. Gradient Boosting (XGBoost/LightGBM)

Instead of training trees independently (like Random Forest), boosting
trains trees sequentially — each new tree focuses on correcting the errors
of the previous ones.

```python
import xgboost as xgb

model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4)
model.fit(X_train, y_train)
```

Boosting usually beats bagging on accuracy but is more prone to overfitting
if `n_estimators`/`learning_rate` aren't tuned — watch your validation score.

---

## 3. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {"n_estimators": [100, 200, 300], "max_depth": [4, 6, 8]}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="f1")
grid.fit(X_train, y_train)
grid.best_params_
```

`GridSearchCV` tries every combination (thorough, slow); `RandomizedSearchCV`
samples a fixed number of combinations (faster, good for large search spaces).

---

## 4. Feature Importance

```python
importances = rf.feature_importances_
pd.Series(importances, index=X_train.columns).sort_values().plot(kind="barh")
```

This tells you which features actually drove predictions — useful both for
model debugging and for explaining results to a non-technical audience.

---

## Project
Take last week's dataset and beat your previous best score:
- [ ] Train a Random Forest and a Gradient Boosting model
- [ ] Tune hyperparameters with GridSearchCV or RandomizedSearchCV
- [ ] Plot feature importances
- [ ] Document the improvement over Week 5's best model

## Resources
- StatQuest: Random Forests — https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
- StatQuest: Gradient Boost — https://www.youtube.com/watch?v=3CC4N4z3GJc
- XGBoost docs — https://xgboost.readthedocs.io/
