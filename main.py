import pandas as pd
import os
import numpy as np

from brats.load_data import load_data
from preprocessing import imputation
from preprocessing import robust_scaler
from split import split_train_test
from PCA import PCA

print('lets go')
data = load_data()
data_train, data_test, labels_train, labels_test = split_train_test(data)
#scaled_data = robust_scaler(data_train)
pca_data = PCA(data_train)
#imputed_data = imputation(data_train)
#print(imputed_data)

print(data_train[1])

