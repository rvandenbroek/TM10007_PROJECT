# Preprocessing

from sklearn import preprocessing
import pandas as pd
import numpy as np 

def robust_scaler(input_data):
    """This function scales the features in a robust way """
    scaler = preprocessing.RobustScaler()
    scaler.fit(input_data)
    scaled_data = scaler.transform(input_data)
    
    return (scaled_data)

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# # load dataset into Pandas DataFrame
# df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width', 'target'])
# df.pop('target')
# input_data = np.array(df)
# print(input_data)

# scaled_data = robust_scaler(input_data)
# print(scaled_data)