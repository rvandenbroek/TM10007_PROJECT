import math
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import RepeatedKFold
from brats.load_data import load_data
from preprocessing import imputation
from preprocessing import robust_scaler
from split import split_train_test
from PCA import PCA_algorithm
from cross_validation import rfe
from preprocessing import dropnan
from k_fold_cross_val import k_fold_cross_validation

from KNN import knn_classifier
from SVM import SVM_algorithm
from SVM import SVM_hyper
from SVM import SVM_PCA

print('start')
data = load_data()
threshold = math.floor(len(data)*0.5)
data = dropnan(data, threshold)
data_train, data_test, labels_train, labels_test = split_train_test(data)

imputed_train, imputed_test = imputation(data_train, data_test)
scaled_train, scaled_test = robust_scaler(imputed_train, imputed_test)
print('done')

#X_train_pca, X_test_pca = PCA_algorithm(scaled_train, scaled_test)
#X = knn_classifier(scaled_train, labels_train, scaled_test, labels_test)
#print(X)

#X_train_pca, X_test_pca = PCA_algorithm(scaled_train, scaled_test)
#Y = SVM_hyper(X_train_pca, labels_train, X_test_pca, labels_test)
#print(Y)

Y = SVM_PCA(scaled_train, labels_train, scaled_test, labels_test)