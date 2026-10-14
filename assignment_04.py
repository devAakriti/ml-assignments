# Dataset: built into scikit-learn (Breast Cancer Wisconsin, 569 samples)
# Also available at https://archive.ics.uci.edu/dataset/17
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

# 1. Load data
data = load_breast_cancer()
X, y = data.data, data.target
print("Shape:", X.shape)
print("Classes:", data.target_names)

# 2. Train-test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Train Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
print("Class priors:", nb.class_prior_)

# 4. Predict and evaluate
y_pred = nb.predict(X_test)
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1-score :", round(f1_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 5. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
ConfusionMatrixDisplay(cm, display_labels=data.target_names).plot(cmap="Greys")
plt.title("Confusion Matrix - Gaussian Naive Bayes")
plt.show()

# 6. Posterior probabilities for the first 5 test samples
print(np.round(nb.predict_proba(X_test[:5]), 4))

# 7. 10-fold cross validation
scores = cross_val_score(GaussianNB(), X, y, cv=10)
print("CV accuracy:", round(scores.mean(), 4), "+/-", round(scores.std(), 4))

# 8. Multinomial Naive Bayes on a small text example
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
texts = ["win money now", "free prize win", "meeting at noon",
         "project report attached", "claim free money", "lunch tomorrow"]
labels = [1, 1, 0, 0, 1, 0]          # 1 = spam, 0 = not spam
vec = CountVectorizer()
mnb = MultinomialNB(alpha=1.0).fit(vec.fit_transform(texts), labels)
test = vec.transform(["free money prize", "report for the meeting"])
print("Spam prediction:", mnb.predict(test))
