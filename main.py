import pandas as pd
import os
import numpy as np
import math
from brats.load_data import load_data
from preprocessing import imputation
from preprocessing import robust_scaler
from split import split_train_test
from PCA import PCA_algorithm
#from cross_validation import RFE

data = load_data()
threshold = math.floor(len(data)*0.5)  
data_drop = data.dropna(thresh=threshold, axis=1)
data_train, data_test, labels_train, labels_test = split_train_test(data_drop)

#data_train=data_train.astype(float)
#data_train = list(pd.to_numeric(list(data_train), errors='coerce'))

data_train = data_train.replace(r'#DIV/0!', np.nan, regex=True)
data_train = data_train.replace([np.inf, -np.inf], np.nan, regex=True)

imputed_data = imputation(data_train)
print(f'imputed data={imputed_data}')
scaled_data = robust_scaler(imputed_data)
print(f'scaled data = {scaled_data}')

