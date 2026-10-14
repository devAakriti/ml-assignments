# Dataset: synthetic noisy sine data generated in the code (no download)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create a small noisy dataset
np.random.seed(0)
X = np.sort(np.random.rand(40, 1) * 6, axis=0)
y = np.sin(X).ravel() + 0.3 * np.random.randn(40)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

# 2. High degree polynomial models: without and with regularization
degree = 12
def make_model(reg):
    return make_pipeline(PolynomialFeatures(degree, include_bias=False),
                         StandardScaler(), reg)

models = {
    "No regularization": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.01, max_iter=100000),
    "Elastic Net": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=100000),
}

# 3. Train and compare train / test performance
X_plot = np.linspace(0, 6, 300).reshape(-1, 1)
plt.figure(figsize=(9, 6))
plt.scatter(X_train, y_train, color="black", label="Train data")
plt.scatter(X_test, y_test, color="gray", marker="s", label="Test data")
print(f"{'Model':20s}{'Train R2':>10s}{'Test R2':>10s}{'Test MSE':>10s}")
for name, reg in models.items():
    pipe = make_model(reg).fit(X_train, y_train)
    tr = r2_score(y_train, pipe.predict(X_train))
    te = r2_score(y_test, pipe.predict(X_test))
    mse = mean_squared_error(y_test, pipe.predict(X_test))
    print(f"{name:20s}{tr:10.3f}{te:10.3f}{mse:10.3f}")
    plt.plot(X_plot, pipe.predict(X_plot), label=name)
plt.ylim(-2.5, 2.5)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Effect of regularization on a degree-12 polynomial")
plt.legend()
plt.show()

# 4. Compare the size of the coefficients
for name in ["No regularization", "Ridge (L2)", "Lasso (L1)"]:
    pipe = make_model(models[name]).fit(X_train, y_train)
    coef = pipe[-1].coef_
    print(f"{name:20s} max |coef| = {np.abs(coef).max():10.3f}   "
          f"zero coefficients = {(np.abs(coef) < 1e-8).sum()}")

# 5. Choose the best alpha using cross-validation
ridge_cv = make_model(RidgeCV(alphas=np.logspace(-3, 3, 50))).fit(X_train, y_train)
lasso_cv = make_model(LassoCV(alphas=np.logspace(-4, 0, 50), cv=5,
                              max_iter=100000)).fit(X_train, y_train)
print("Best Ridge alpha:", ridge_cv[-1].alpha_)
print("Best Lasso alpha:", lasso_cv[-1].alpha_)
print("Ridge CV test R2:", round(r2_score(y_test, ridge_cv.predict(X_test)), 3))
print("Lasso CV test R2:", round(r2_score(y_test, lasso_cv.predict(X_test)), 3))
