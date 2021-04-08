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


def PCA_algorithm(X_train_scaled, X_test_scaled):
    pca = decomposition.PCA(n_components=10)
    pca.fit(X_train_scaled)
    variance = pca.explained_variance_ratio_
    cumvariance = variance.cumsum()
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca, X_test_pca

# Stukje code voor het maken van boxplots

# print('start')
# data = load_data()
# threshold = math.floor(len(data)*0.5)
# data = dropnan(data, threshold)
# data_train, data_test, labels_train, labels_test = split_train_test(data)
# imputed_train, imputed_test = imputation(data_train, data_test)
# scaled_train, scaled_test = robust_scaler(imputed_train, imputed_test)
# # print(f'scaled: {(scaled_train.shape)}')
# X_train_pca, X_test_pca = PCA_algorithm(scaled_train, scaled_test)
# print(f'X_train: {(X_train_pca.shape)}')

# names = ['PC 1', 'PC 2', 'PC 3']
# X_train_df = pd.DataFrame(data=X_train_pca, columns=names)
# X_train_df['labels'] = labels_train
# X_train_df1 = X_train_df[X_train_df['labels'] == 'GBM']
# X_train_df2 = X_train_df[X_train_df['labels'] == 'LGG']

# fig, axs = plt.subplots(1, 3)
# axs[0].boxplot(x=X_train_df1['PC 1'])
# axs[0].set_title('PC 1 GBM')
# axs[1].boxplot(x=X_train_df1['PC 2'])
# axs[1].set_title('PC 2 GBM')
# axs[2].boxplot(x=X_train_df1['PC 3'])
# axs[2].set_title('PC 3 GBM')
# plt.show()

# fig, axs = plt.subplots(1, 3)
# axs[0].boxplot(x=X_train_df2['PC 1'])
# axs[0].set_title('PC 1 LGG')
# axs[1].boxplot(x=X_train_df2['PC 2'])
# axs[1].set_title('PC 2 LGG')
# axs[2].boxplot(x=X_train_df2['PC 3'])
# axs[2].set_title('PC 3 LGG')
# plt.show()

# Dimensionality reduction
# Class visualization 