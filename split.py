# split into train and test set

from brats.load_data import load_data
from sklearn import model_selection
import numpy as np

data = load_data()
#print(f'The number of samples: {len(data.index)}')
#print(f'The number of columns: {len(data.columns)}')

def split_train_test(data):
    '''This function will split the data in train and test'''
    labels = np.array(data['label'])
    data.pop('label')
    data_array = np.array(data[:])

    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    # Later percentages veranderen en kijken naar de uitkomst: dit ook onderbouwen
    # Shuffle default = True 
    data_train, data_test, labels_train, labels_test = model_selection.train_test_split(data, labels, stratify=labels, train_size=0.8)

    return data_train, data_test, labels_train, labels_test
