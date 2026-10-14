# Dataset: synthetic data generated in the code (no download required)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Generate data: y = 4 + 3x + noise
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 2. Linear regression using the Normal Equation
Xb_train = np.c_[np.ones((len(X_train), 1)), X_train]
Xb_test = np.c_[np.ones((len(X_test), 1)), X_test]
theta_ne = np.linalg.inv(Xb_train.T @ Xb_train) @ Xb_train.T @ y_train
print("Normal equation  : intercept = %.4f, slope = %.4f"
      % (theta_ne[0, 0], theta_ne[1, 0]))

# 3. Linear regression using Gradient Descent (implemented from scratch)
def gradient_descent(Xb, y, lr=0.1, epochs=1000):
    m = len(y)
    theta = np.zeros((Xb.shape[1], 1))
    cost_history = []
    for _ in range(epochs):
        error = Xb @ theta - y
        gradient = (1 / m) * Xb.T @ error
        theta = theta - lr * gradient
        cost_history.append((1 / (2 * m)) * np.sum(error ** 2))
    return theta, cost_history

theta_gd, costs = gradient_descent(Xb_train, y_train)
print("Gradient descent : intercept = %.4f, slope = %.4f"
      % (theta_gd[0, 0], theta_gd[1, 0]))

# 4. Linear regression using scikit-learn
model = LinearRegression().fit(X_train, y_train)
print("Scikit-learn     : intercept = %.4f, slope = %.4f"
      % (model.intercept_[0], model.coef_[0, 0]))

# 5. Predictions and evaluation on the test set
y_pred = Xb_test @ theta_gd
print("MAE :", round(mean_absolute_error(y_test, y_pred), 4))
print("MSE :", round(mean_squared_error(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
print("R2  :", round(r2_score(y_test, y_pred), 4))

# 6. Plot the regression line and the cost curve
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(X_train, y_train, color="black", s=15, label="Train")
ax[0].scatter(X_test, y_test, color="gray", s=25, label="Test")
line = np.array([[0], [2]])
ax[0].plot(line, theta_gd[0] + theta_gd[1] * line, "k-", label="Fitted line")
ax[0].set_xlabel("X")
ax[0].set_ylabel("y")
ax[0].set_title("Linear Regression Fit")
ax[0].legend()
ax[1].plot(costs, "k-")
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Cost J")
ax[1].set_title("Gradient Descent Convergence")
plt.show()

# 7. Predict a new value
x_new = 1.5
print("Prediction for x = 1.5:", round(float(theta_gd[0, 0] + theta_gd[1, 0] * x_new), 4))
