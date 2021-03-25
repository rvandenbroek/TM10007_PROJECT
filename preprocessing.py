# Preprocessing

from sklearn import preprocessing
import pandas as pd
import numpy as np 
from sklearn.impute import KNNImputer

def robust_scaler(input_data):
    """This function scales the features in a robust way """
    scaler = preprocessing.RobustScaler()
    scaler.fit(input_data)
    scaled_data = scaler.transform(input_data)
    
    return (scaled_data)

def imputation(data_frame):
    '''With this function, the missing features are imputed with KNN'''
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    imputed = imputer.fit_transform(data_frame)

    return imputed


nan = np.nan
X = [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]]


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])
df.pop('target')
input_data = X
imputed_data = imputation(input_data)
scaled_data = robust_scaler(imputed_data)
print(scaled_data)