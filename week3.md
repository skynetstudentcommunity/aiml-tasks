# Week 3: Data Wrangling & Feature Engineering — Tutorial

## Why this week matters
Real data is messy: missing values, inconsistent categories, wildly
different scales between columns. Models don't handle this gracefully on
their own — you have to clean and shape the data first. This is often 70%+
of real-world ML work.

---

## 1. Missing Values

Two options: drop or impute (fill in).

```python
import pandas as pd
df = pd.read_csv("messy.csv")

df.isnull().sum()                     # count missing per column

df_dropped = df.dropna()              # drop any row with a missing value
df_filled = df.fillna(df.mean(numeric_only=True))  # fill numeric cols with mean
df["category_col"] = df["category_col"].fillna("Unknown")
```

**Rule of thumb:** drop if very few rows are affected and losing them won't
bias your data; impute if the column matters and you have a sensible way to
fill it (mean/median for numeric, mode or "Unknown" for categorical).

---

## 2. Encoding Categorical Variables

Models need numbers, not text categories.

```python
# One-hot encoding: each category becomes its own 0/1 column
pd.get_dummies(df, columns=["color"])

# Label encoding: each category becomes an integer (use for ordinal data,
# e.g. "low"/"medium"/"high" — order matters)
from sklearn.preprocessing import LabelEncoder
df["size_encoded"] = LabelEncoder().fit_transform(df["size"])
```

Use one-hot for unordered categories (colors, cities). Use label encoding
only when there's a natural order — otherwise you're implying a false
ranking (e.g. "red=0, blue=1, green=2" has no real meaning).

---

## 3. Feature Scaling

Many algorithms (KNN, gradient descent–based models, PCA) are sensitive to
the scale of features. A column ranging 0–1,000,000 will dominate a column
ranging 0–1 unless you scale them.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization: mean 0, std 1 — good default, handles outliers okay
X_scaled = StandardScaler().fit_transform(df[["age", "income"]])

# Normalization: squashes to [0, 1] range — good when you need bounded values
X_norm = MinMaxScaler().fit_transform(df[["age", "income"]])
```

---

## 4. Train/Validation/Test Splits

Never evaluate a model on the same data it learned from — that's how you
fool yourself into thinking a model is good.

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```

- **Train**: what the model learns from
- **Validation**: what you tune hyperparameters against
- **Test**: touched only once, at the very end, to report final performance

---

## Project
Take a messy real dataset (Kaggle has several tagged "messy") and write a
full cleaning pipeline script:
- [ ] Handle missing values with a documented reason for drop vs. impute
- [ ] Encode categorical columns appropriately
- [ ] Scale numeric features
- [ ] Split into train/val/test and save each as a CSV

## Resources
- Kaggle Learn: Data Cleaning — https://www.kaggle.com/learn/data-cleaning
- Kaggle Learn: Feature Engineering — https://www.kaggle.com/learn/feature-engineering
- Scikit-learn preprocessing guide — https://scikit-learn.org/stable/modules/preprocessing.html
