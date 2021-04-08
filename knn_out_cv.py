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
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from Random_forest import RF_hyperpara, random_forest_algoritm

print('start')
data = load_data()
threshold = math.floor(len(data)*0.5)
data = dropnan(data, threshold)
labels = np.array(data['label'])
data.pop('label')


