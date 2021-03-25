# General packages
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
import seaborn

from sklearn import model_selection
from sklearn import metrics
from sklearn import feature_selection 
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import svm

# Other scalers 
# preprocessing.MinMaxScaler() --> scale from 0 1 
# preprocessing.MaxAbsScaler() --> -1 to 1
# RobustScaler as a drop-in replacement
# RobustScaler(*, with_centering=True, with_scaling=True, quantile_range=25.0, 75.0, copy=True, unit_variance=False)


# Scale the dataset
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


clf_knn = neighbors.KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(X_scaled, y)

# Test the classifier on the training data and plot
score_train = clf_knn.score(X_scaled, y)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_title(f"Training performance: accuracy {score_train}")
colorplot(clf_knn, ax, X_scaled[:, 0],X_scaled[:, 1], h=1000)
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], marker='o', c=y,
           s=25, edgecolor='k', cmap=plt.cm.Paired)