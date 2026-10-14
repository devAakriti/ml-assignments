# Dataset: built into scikit-learn (Iris dataset, 150 samples, 4 features)
# Also available at https://archive.ics.uci.edu/dataset/53/iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target
print("Features:", iris.feature_names)
print("Classes :", iris.target_names)

# 2. Split and scale the features
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 3. Find the best K using 5-fold cross validation
k_values = range(1, 26)
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores.append(cross_val_score(knn, X_train_s, y_train, cv=5).mean())

best_k = k_values[int(np.argmax(cv_scores))]
print("Best K:", best_k, "with CV accuracy:", round(max(cv_scores), 4))

plt.figure(figsize=(7, 4))
plt.plot(k_values, cv_scores, "ko-")
plt.xlabel("Value of K")
plt.ylabel("Cross-validated accuracy")
plt.title("Selecting K for KNN")
plt.grid(True)
plt.show()

# 4. Train the final model with the best K
knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
knn.fit(X_train_s, y_train)
y_pred = knn.predict(X_test_s)

# 5. Evaluation
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, target_names=iris.target_names))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot(cmap="Greys")
plt.title("Confusion Matrix - KNN")
plt.show()

# 6. Predict the class of a new flower
new_sample = scaler.transform([[5.1, 3.5, 1.4, 0.2]])
pred = knn.predict(new_sample)
print("Predicted class:", iris.target_names[pred[0]])
print("Class probabilities:", knn.predict_proba(new_sample))

# 7. Compare distance metrics
for metric in ["euclidean", "manhattan", "chebyshev"]:
    model = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
    model.fit(X_train_s, y_train)
    print(metric, "accuracy:", round(model.score(X_test_s, y_test), 4))
