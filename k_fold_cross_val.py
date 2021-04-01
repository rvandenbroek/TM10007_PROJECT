import numpy as np
from sklearn.model_selection import RepeatedKFold
from main import data_train, data_test, labels_train, labels_test
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])


n_splits_hyper_para = 3
rkf = RepeatedKFold(n_splits=n_splits_hyper_para, n_repeats=n_splits_hyper_para)
for train_index, test_index in rkf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]