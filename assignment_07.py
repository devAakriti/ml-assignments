# Dataset: synthetic noisy sine wave generated in the code (no download)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Generate non-linear data
np.random.seed(1)
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel() + 0.2 * np.random.randn(200)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# 2. Train SVR with three different kernels
kernels = {
    "Linear": SVR(kernel="linear", C=100, epsilon=0.1),
    "Polynomial": SVR(kernel="poly", C=100, degree=3, epsilon=0.1),
    "RBF": SVR(kernel="rbf", C=100, gamma=0.5, epsilon=0.1),
}
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color="gray", s=15, label="Training data")
for name, svr in kernels.items():
    svr.fit(X_train, y_train)
    pred = svr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name:10s} RMSE = {rmse:.4f}  R2 = {r2_score(y_test, pred):.4f}"
          f"  Support vectors = {len(svr.support_)}")
    plt.plot(X, svr.predict(X), label=name + " kernel")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Support Vector Regression with different kernels")
plt.legend()
plt.show()

# 3. Hyperparameter tuning with GridSearchCV (scaled inputs)
pipe = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
param_grid = {
    "svr__C": [0.1, 1, 10, 100],
    "svr__gamma": [0.01, 0.1, 1],
    "svr__epsilon": [0.01, 0.1, 0.2],
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_mean_squared_error")
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)

# 4. Evaluate the best model
best = grid.best_estimator_
y_pred = best.predict(X_test)
print("MAE :", round(mean_absolute_error(y_test, y_pred), 4))
print("MSE :", round(mean_squared_error(y_test, y_pred), 4))
print("R2  :", round(r2_score(y_test, y_pred), 4))

# 5. Plot the tuned model with the epsilon tube
eps = best.named_steps["svr"].epsilon
y_line = best.predict(X)
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color="gray", s=15, label="Training data")
plt.plot(X, y_line, "k-", label="Tuned SVR (RBF)")
plt.plot(X, y_line + eps, "k--", label="Epsilon tube")
plt.plot(X, y_line - eps, "k--")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Tuned SVR with epsilon-insensitive tube")
plt.legend()
plt.show()
