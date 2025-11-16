import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.701632/full
# data preprocessing steps: bandpass filter (2-10hz), z score normalization, 80/20 trian test split or use 10 fold cross validation


# data here:
X, y = 1,2 # Replace with actual data loading or generation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scikit learn implementation: https://www.geeksforgeeks.org/machine-learning/ml-non-linear-svm

svm = SVC(kernel='rbf', C=2, gamma=0.01)  # RBF kernel allows learning circular boundaries
svm.fit(X_train, y_train)

# predictions
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")