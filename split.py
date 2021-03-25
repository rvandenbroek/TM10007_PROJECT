# Data loading functions. Uncomment the one you want to use
#from adni.load_data import load_data
from brats.load_data import load_data
from sklearn import model_selection
#from hn.load_data import load_data
#from ecg.load_data import load_data
import numpy as np

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

keys = np.array(data.keys())
#print(keys)
y = np.array(data['label'])
data.pop('label')
X = np.array(data[:])

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# Later percentages veranderen en kijken naar de uitkomst: dit ook onderbouwen
# Shuffle default = True 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.85)

np.count_nonzero(np.isnan(data['VOLUME_NET_OVER_ED']))
