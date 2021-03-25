# Scaler


import numpy as np 
import pandas as pd
from sklearn import preprocessing

def robust_scaler(input_data):
    """This function scales the features in a robust way """
    scaler = preprocessing.RobustScaler()
    scaler.fit(input_data)
    scaled_data = scaler.transform(input_data)
    
    return (scaled_data)
