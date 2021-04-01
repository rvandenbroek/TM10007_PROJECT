import math
import pandas as pd
import os
import numpy as np
from brats.load_data import load_data
from preprocessing import imputation
from preprocessing import robust_scaler
from split import split_train_test
from PCA import PCA_algorithm
from cross_validation import rfe
from preprocessing import dropnan

print('start')
data = load_data()
threshold = math.floor(len(data)*0.5)
data = dropnan(data, threshold)
data_train, data_test, labels_train, labels_test = split_train_test(data)

imputed_train, imputed_test = imputation(data_train, data_test)
scaled_train, scaled_test = robust_scaler(imputed_train, imputed_test)
print('done')
