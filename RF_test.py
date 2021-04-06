import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
import seaborn
from main import scaled_train, labels_train, scaled_test, labels_test
# Classifiers
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
#k_list = list(range(1, 26, 2))

def random_forest_class(scaled_train, labels_train, scaled_test, labels_test, max_trees=100, show_fig=False):
    n_trees = list(range(1,max_trees,5))
    all_train = []
    all_test = []
    X2 = scaled_train
    y2 = labels_train
    #    Repeat the experiment 20 times, use 20 random splits in which class balance is retained
    sss = model_selection.StratifiedShuffleSplit(n_splits=20, test_size=0.5, random_state=None)
    #print(sss)
    for train_index, test_index in sss.split(X2, y2):
        train_scores = []
        test_scores = []
        
        split_X_train = X2[train_index]
        split_y_train = y2[train_index]
        split_X_test = X2[test_index]
        split_y_test = y2[test_index]

        for n in n_trees:
            clf_RF = RandomForestClassifier(n_estimators=n)
            #print(clf_RF)
            #clf_knn = neighbors.KNeighborsClassifier(n_neighbors=k)
            
            clf_RF.fit(split_X_train, split_y_train)

            # Test the classifier on the training data and plot
            score_train = clf_RF.score(split_X_train, split_y_train)
            score_test = clf_RF.score(split_X_test, split_y_test)

            train_scores.append(score_train)
            test_scores.append(score_test)
            
        all_train.append(train_scores)
        all_test.append(test_scores)
        

    # Create numpy array of scores and calculate the mean and std
    all_train = np.array(all_train)
    all_test = np.array(all_test)

    train_scores_mean = all_train.mean(axis=0)
    train_scores_std = all_train.std(axis=0)

    test_scores_mean = all_test.mean(axis=0)
    test_scores_std = all_test.std(axis=0)

    if show_fig:
        # Plot the mean scores and the std as shading
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.grid()
        ax.fill_between(n_trees, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
        ax.fill_between(n_trees, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1,
                            color="g")
        ax.plot(n_trees, train_scores_mean, 'o-', color="r",
                label="Training score")
        ax.plot(n_trees, test_scores_mean, 'o-', color="g",
                label="Test score")
        plt.show()
    return (train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)