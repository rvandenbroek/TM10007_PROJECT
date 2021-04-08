import math
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import RepeatedKFold
from brats.load_data import load_data
from preprocessing import imputation
from preprocessing import robust_scaler
from preprocessing import standard_scaler
from split import split_train_test
from PCA import PCA_algorithm
from cross_validation import rfe
from preprocessing import dropnan
from k_fold_cross_val import k_fold_cross_validation
from KNN import knn_classifier
from SVM import SVM_algorithm
from SVM import SVM_hyper
from SVM import SVM_PCA
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from Random_forest import RF_hyperpara, random_forest_algoritm
from sklearn import neighbors
import seaborn
import matplotlib.pyplot as plt

print('start')
data = load_data()
threshold = math.floor(len(data)*0.5)
data = dropnan(data, threshold)
labels = np.array(data['label'])
data.pop('label')

# Create a 20 fold stratified CV iterator
cv_20fold = model_selection.StratifiedKFold(n_splits=10)
results = []
best_n_trees = []

# Loop over the folds
for validation_index, test_index in cv_20fold.split(data, labels):
    # Split the data properly
    X_validation = data.iloc[validation_index]
    y_validation = labels[validation_index]
    
    X_test = data.iloc[test_index]
    y_test = labels[test_index]

    imputed_train, imputed_test = imputation(X_validation, X_test)
    scaled_train, scaled_test = robust_scaler(imputed_train, imputed_test)
    pca_train, pca_test = PCA_algorithm(scaled_train, scaled_test)
    #pca_train, pca_test = robust_scaler(imputed_train, imputed_test)

    # Create a grid search to find the optimal k using a gridsearch and 10-fold cross validation
    # Same as above
    parameters = {"n_estimators": list(range(20, 40, 2))}
    RF = RandomForestClassifier()
    cv_10fold = model_selection.StratifiedKFold(n_splits=10)
    grid_search = model_selection.GridSearchCV(RF, parameters, cv=cv_10fold, scoring='roc_auc')
    grid_search.fit(pca_train, y_validation)
    
    # Get resulting classifier
    clf = grid_search.best_estimator_
    print(f'Best trees: n={clf.n_estimators}')
    best_n_trees.append(clf.n_estimators)
    
    # Test the classifier on the test data
    probabilities = clf.predict_proba(pca_test)
    scores = probabilities[:, 1]
    
    # Get the auc
    auc = metrics.roc_auc_score(y_test, scores)
    results.append({
        'auc': auc,
        'n': clf.n_estimators,
        'set': 'test'
    })
    
    # Test the classifier on the validation data
    probabilities_validation = clf.predict_proba(pca_train)
    scores_validation = probabilities_validation[:, 1]
    
    # Get the auc
    auc_validation = metrics.roc_auc_score(y_validation, scores_validation)
    results.append({
        'auc': auc_validation,
        'n': clf.n_estimators,
        'set': 'validation'
    })
    
# Create results dataframe and plot it
results = pd.DataFrame(results)
seaborn.boxplot(y='auc', x='set', data=results)
plt.show()
optimal_n = int(np.median(best_n_trees))
print(f"The optimal N={optimal_n}")
