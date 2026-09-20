# Week 1: Python for Data — Tutorial

## Why this week matters
Every week after this assumes you can load data, manipulate it, and look at it.
NumPy, Pandas, and Matplotlib/Seaborn are the "alphabet" of ML in Python — skipping
them makes everything downstream harder than it needs to be.

---

## 1. Setup
Create an isolated environment so packages don't conflict across projects:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install numpy pandas matplotlib seaborn jupyter
```

Open a notebook with `jupyter notebook` or use the Jupyter extension in VS Code.
Notebooks let you run code in chunks and see plots inline — much better for
exploration than a plain `.py` script.

---

## 2. NumPy — arrays and vectorized operations

NumPy's core object is the `ndarray`. The big idea: operations happen on whole
arrays at once ("vectorized"), which is both faster and more readable than
writing loops.

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

a + b        # array([11, 22, 33, 44]) — no loop needed
a * 2        # array([2, 4, 6, 8])
a.mean()     # 2.5
a[a > 2]     # array([3, 4])  <- boolean indexing, used constantly in ML code
```

**Broadcasting** lets NumPy apply an operation between arrays of different
shapes without you writing explicit loops:

```python
matrix = np.array([[1, 2], [3, 4], [5, 6]])   # shape (3, 2)
matrix + np.array([10, 100])                   # adds [10,100] to every row
```

Why it matters for ML: under the hood, features are stored as matrices, and
model math (weighted sums, gradients) is just array operations like these.

---

## 3. Pandas — DataFrames

A DataFrame is a table: rows are observations, columns are features.

```python
import pandas as pd

df = pd.read_csv("titanic.csv")
df.head()                     # first 5 rows
df.info()                     # column types, missing values
df.describe()                 # summary stats for numeric columns

df["Age"].mean()
df[df["Survived"] == 1]       # filter rows
df.groupby("Pclass")["Fare"].mean()   # average fare per passenger class
df.merge(other_df, on="PassengerId") # combine two tables
```

**Mental model:** almost every Pandas operation is "select rows/columns,
apply a function, combine the result" — `filter → apply → combine` is the
pattern you'll reuse constantly (it's literally what `groupby` does).

---

## 4. Visualization — Matplotlib & Seaborn

Matplotlib is the low-level plotting library; Seaborn sits on top of it and
makes statistical plots easier and prettier.

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df["Age"].dropna(), bins=30)
plt.title("Age Distribution")
plt.show()

sns.boxplot(x="Pclass", y="Fare", data=df)
sns.scatterplot(x="Age", y="Fare", hue="Survived", data=df)
sns.heatmap(df.corr(numeric_only=True), annot=True)   # correlation matrix
```

**Rule of thumb for choosing a plot:**
- One numeric variable → histogram
- One numeric vs. one categorical → boxplot or violin plot
- Two numeric variables → scatter plot
- Many numeric variables at once → correlation heatmap

---

## 5. Putting it together: a mini EDA workflow

1. `df.info()` / `df.describe()` — get the lay of the land
2. Check for missing values: `df.isnull().sum()`
3. Plot distributions of key numeric columns
4. Plot relationships between features and your target variable
5. Write down 3–5 observations in plain English (this is what actually gets
   graded/read by anyone reviewing your portfolio — code without insight is
   just code)

---

## Project
Load a public CSV dataset (Titanic or your choice). Produce a notebook with:
- [ ] `df.info()`, `df.describe()`, missing-value check
- [ ] 5+ visualizations (mix of histogram, boxplot, scatter, heatmap)
- [ ] Written observations under each plot — what does it tell you?

## Resources
- NumPy Quickstart — https://numpy.org/doc/stable/user/quickstart.html
- Pandas "10 Minutes to pandas" — https://pandas.pydata.org/docs/user_guide/10min.html
- Kaggle Learn: Pandas (free micro-course) — https://www.kaggle.com/learn/pandas
- Seaborn tutorial — https://seaborn.pydata.org/tutorial.html
- Dataset: Titanic (Kaggle) — https://www.kaggle.com/c/titanic
