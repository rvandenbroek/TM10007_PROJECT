# Preprocessing

from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def dropnan(data, threshold):
    '''
    First of all, cetrain strings and values in the dataset are turned into NaN.
    With this function, features with more than a certain amount of NaN values will be
    excluded from the dataset. The inputs are the contaminated dataset and the
    treshold value. The output is the cleaned dataset. '''
    # replace '#DIV/0!' and infinity values with NaN
    data = data.replace(r'#DIV/0!', np.nan, regex=True)
    data = data.replace([np.inf, -np.inf], np.nan, regex=True)
    # drop the features with more than threshold NaN's
    data = data.dropna(thresh=threshold, axis=1)
    print(data.shape)
    return data


def imputation(train_data, test_data):
    ''' In order to substitute the NaN values rather then delete them, a kNN imputer function is
    used to impute the missing data. This function is based on the train set and subsequently
    applied on the test set. This ensures the model is completely trained on the train set rather than
    the test set.
    The inputs are the trainset and the testset, the outputs are the same sets with imputed values.
    '''
    # impute the still existing NaN's
    imputer = KNNImputer(n_neighbors=3, weights="uniform")
    imputed_train = imputer.fit_transform(train_data)
    imputed_test = imputer.transform(test_data)

    return imputed_train, imputed_test


def robust_scaler(train_data, test_data):
    """This scaler removes the median and scales the data according to the quantile range.
    The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).
    The scaler is fit on the train dataset and applied to the test data set to prevent training on the test set. """
    scaler = preprocessing.RobustScaler()
    scaler.fit(train_data)
    scaled_train = scaler.transform(train_data)
    scaled_test = scaler.transform(test_data)

    return (scaled_train, scaled_test)
