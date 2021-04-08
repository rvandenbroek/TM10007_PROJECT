# Preprocessing

from sklearn import preprocessing
import pandas as pd
import numpy as np 
from sklearn.impute import KNNImputer


def dropnan(data, threshold):
    '''With this function, features with more than threshold amount of NaN's will be dropped.'''
    # replace '#DIV/0!' and infinity values with NaN
    data = data.replace(r'#DIV/0!', np.nan, regex=True)
    data = data.replace([np.inf, -np.inf], np.nan, regex=True)
    # drop the features with more than threshold NaN's
    data = data.dropna(thresh=threshold, axis=1)
    print(data.shape)
    return data


def imputation(train_data, test_data):
    '''With this function, the missing feature values are imputed with KNN'''
    # impute the still existing NaN's 
    imputer = KNNImputer(n_neighbors=3, weights="uniform")
    imputed_train = imputer.fit_transform(train_data)
    imputed_test = imputer.transform(test_data)

    return imputed_train, imputed_test


def robust_scaler(train_data, test_data):
    """This function scales the features in a robust way """
    scaler = preprocessing.RobustScaler()
    scaler.fit(train_data)
    scaled_train = scaler.transform(train_data)
    scaled_test = scaler.transform(test_data)

    return (scaled_train, scaled_test)

# nan = np.nan
# X = [[1, '21', nan], [0, 40, 3], [nan, 6, 5], [8, 8, 7]]
# print(type(X))


# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# # load dataset into Pandas DataFrame
# df = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])
# df.pop('target')
# input_data = X
# imputed_data = imputation(input_data)
# scaled_data = robust_scaler(imputed_data)
# print(scaled_data)