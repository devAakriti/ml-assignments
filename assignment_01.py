# Dataset: https://www.kaggle.com/datasets/ashydv/advertising-dataset
# Download advertising.csv and keep it in the working directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

# 1. Load and inspect the data
df = pd.read_csv("advertising.csv")
print(df.head())
print(df.info())
print(df.describe())

# Independent variables: TV, Radio, Newspaper | Dependent variable: Sales
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# 2. Correlation matrix and heatmap
corr = df.corr()
print(corr)
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="gray_r", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# 3. Scatter plot of each independent variable against Sales
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, X.columns):
    ax.scatter(df[col], y, color="black", s=12)
    ax.set_xlabel(col)
    ax.set_ylabel("Sales")
plt.tight_layout()
plt.show()

# 4. Pearson correlation and p-value (hypothesis test, H0: r = 0)
for col in X.columns:
    r, p = stats.pearsonr(df[col], y)
    print(f"{col}: r = {r:.3f}, p-value = {p:.4g}")

# 5. Simple linear regression for each predictor
for col in X.columns:
    res = stats.linregress(df[col], y)
    print(f"{col}: slope = {res.slope:.4f}, "
          f"intercept = {res.intercept:.4f}, R2 = {res.rvalue**2:.4f}")

# 6. Multiple linear regression (all predictors together)
model = LinearRegression().fit(X, y)
print("Intercept:", round(model.intercept_, 4))
for name, coef in zip(X.columns, model.coef_):
    print(f"Coefficient of {name}: {coef:.4f}")
print("R-squared:", round(model.score(X, y), 4))

# 7. Check multicollinearity among independent variables
print(X.corr())
