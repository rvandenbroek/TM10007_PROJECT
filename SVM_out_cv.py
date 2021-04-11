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
from sklearn.svm import SVC
from statistics import mean
from statistics import stdev

print('start')
data = load_data()
threshold = math.floor(len(data)*0.5)
data = dropnan(data, threshold)
labels = np.array(data['label'])
data.pop('label')

# Create a 20 fold stratified CV iterator
cv_10fold = model_selection.StratifiedKFold(n_splits=10)
results = []
best_C = []
best_coef0 = []
accuracies = []
conf_matrix_list = []

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

# Loop over the folds
for i, (validation_index, test_index) in enumerate(cv_10fold.split(data, labels)):
    # Split the data properly
    X_validation = data.iloc[validation_index]
    y_validation = labels[validation_index]
    
    X_test = data.iloc[test_index]
    y_test = labels[test_index]

    imputed_train, imputed_test = imputation(X_validation, X_test)
    scaled_train, scaled_test = robust_scaler(imputed_train, imputed_test)
    pca_train, pca_test = PCA_algorithm(scaled_train, scaled_test)
    #pca_train, pca_test = standard_scaler(imputed_train, imputed_test)

    # Create a grid search to find the optimal k using a gridsearch and 10-fold cross validation
    # Same as above
    #, "gamma": ['scale'], "coef0": 20, "C":1 "kernel": ['poly', 'linear'], 
    parameters = {"C": [0.4, 0.6, 0.8, 1, 1.2, 1.4], "coef0": list(range(1, 25, 5))}
    svm = SVC(probability=True, gamma='scale', kernel='poly')
    cv_10fold = model_selection.StratifiedKFold(n_splits=10)
    grid_search = model_selection.GridSearchCV(svm, parameters, cv=cv_10fold, scoring='roc_auc')
    grid_search.fit(pca_train, y_validation)
    
    # Get resulting classifier
    clf = grid_search.best_estimator_
    print(f'Best C: k={clf.C}')
    print(f'Best coef0: k={clf.coef0}')
    best_C.append(clf.C)
    best_coef0.append(clf.coef0)
    
    # Test the classifier on the test data
    predicted = clf.predict(pca_test)
    probabilities = clf.predict_proba(pca_test)
    scores = probabilities[:, 1]

    #ROC curve
    viz =  metrics.plot_roc_curve(clf, pca_test, y_test,
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    # Get the auc
    auc = metrics.roc_auc_score(y_test, scores)
    results.append({
        'auc': auc,
        'C': clf.C,
        'coef0': clf.coef0,
        'set': 'test'
    })
    
    # Accuracies
    accuracy = metrics.accuracy_score(y_test, predicted)
    accuracies.append(accuracy)

    # Confusion matrix
    conf_matrix = metrics.confusion_matrix(y_test, predicted)
    conf_matrix_list.append(conf_matrix)

    # Test the classifier on the validation data
    probabilities_validation = clf.predict_proba(pca_train)
    scores_validation = probabilities_validation[:, 1]
    
    # Get the auc
    auc_validation = metrics.roc_auc_score(y_validation, scores_validation)
    results.append({
        'auc': auc_validation,
        'C': clf.C,
        'coef0': clf.coef0,
        'set': 'validation'
    })
    
# Create results dataframe and plot it
results = pd.DataFrame(results)
#seaborn.boxplot(y='auc', x='set', data=results)
#plt.show()

# Accuracy
mean_accuracy = mean(accuracies)
std_accuracy = stdev(accuracies)
print(mean_accuracy)
print(std_accuracy)

# Confusion matrix
mean_of_conf_matrix = np.mean(conf_matrix_list, axis=0)
conf_matrix_df = pd.DataFrame.from_dict({
        'Actual: GBM': mean_of_conf_matrix[0],
        'Actual: LGG': mean_of_conf_matrix[1]},
orient='index', columns=['Predicted: GBM', 'Predicted: LGG'])
#seaborn.heatmap(conf_matrix_df/np.sum(np.sum(conf_matrix_df)), annot= True, fmt='.2%', cmap='Blues')
#plt.show()

# Plot ROC curves
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating curve SVM")
ax.legend(loc="lower right")
plt.show()
