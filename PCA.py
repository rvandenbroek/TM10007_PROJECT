import math
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from brats.load_data import load_data
from preprocessing import imputation
from preprocessing import robust_scaler
from split import split_train_test
from cross_validation import rfe
from preprocessing import dropnan
from sklearn import decomposition 

def PCA_algorithm(X_train_scaled):
    pca = decomposition.PCA(n_components=4)
    pca.fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    #X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca

print('start')
data = load_data()
threshold = math.floor(len(data)*0.5)
data = dropnan(data, threshold)
data_train, data_test, labels_train, labels_test = split_train_test(data)
imputed_train, imputed_test = imputation(data_train, data_test)
scaled_train, scaled_test = robust_scaler(imputed_train, imputed_test)
X_train_pca = PCA_algorithm(scaled_train)
names = ['PC 1', 'PC 2', 'PC 3', 'PC 4']
X_train_df = pd.DataFrame(data=X_train_pca, columns=names)
X_train_df['labels'] = labels_train
# sns.pairplot(X_train_df, hue='labels')
# plt.show()
X_train_df1 = X_train_df[X_train_df['labels'] =='GBM']
X_train_df2 = X_train_df[X_train_df['labels'] =='LGG']
print(X_train_df1.head())

# fig, axs = plt.subplots(2, 2)
# axs[0, 0].
sns.boxplot(x=X_train_df1['PC 2'])
# axs[0, 0].set_title('PC 1')
# axs[0, 1].plot(x, y, 'tab:orange')
# axs[0, 1].set_title('Axis [0, 1]')
# axs[1, 0].plot(x, -y, 'tab:green')
# axs[1, 0].set_title('Axis [1, 0]')
# axs[1, 1].plot(x, -y, 'tab:red')
# axs[1, 1].set_title('Axis [1, 1]')


# sns.boxplot(x=X_train_df1['PC 3'])
plt.show()

# Dimensionality reduction
# Class visualization 