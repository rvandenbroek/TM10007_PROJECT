import pandas as pd
import os
import numpy as np

from brats.load_data import load_data
from preprocessing import imputation
from preprocessing import robust_scaler
from split import split_train_test

print('lets go')

data = load_data()
data_train, data_test, labels_train, labels_test = split_train_test(data)

#data_train=data_train.astype(float)
#data_train = list(pd.to_numeric(list(data_train), errors='coerce'))
print(data_train)
print('tot hier werkt het')
#data_train= list(data_train)

imputed_data = imputation(data_train)
#scaled_data = robust_scaler(data_train)
#print(scaled_data)

#print(imputed_data)
#print(data_train[1])

#laatste_rij=data_train[1]
#print(laatste_rij)
#print(f'laatste = {laatste_rij[-1]}')
#print(type(laatste_rij[-1]))

print('klaar')