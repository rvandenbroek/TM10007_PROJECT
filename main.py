import pandas as pd
import os
import numpy as np
import math
from brats.load_data import load_data
from preprocessing import imputation
from preprocessing import robust_scaler
from split import split_train_test
from PCA import PCA_algorithm
from cross_validation import cross_validation

print('start')
data = load_data()
data_train, data_test, labels_train, labels_test = split_train_test(data)
threshold = math.floor(len(data)*0.5)
imputed_train, imputed_test = imputation(data_train, data_test, threshold)
scaled_train, scaled_test = robust_scaler(imputed_train, imputed_test)
print('done')
