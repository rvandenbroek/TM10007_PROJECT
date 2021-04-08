import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
import seaborn
from score import scoring
#from main import scaled_train, labels_train, scaled_test, labels_test
# Classifiers
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
#k_list = list(range(1, 26, 2))

#def RF_hyperpara(scaled_train, labels_train, scaled_test, labels_test, max_trees=100, show_fig=False):
def RF_hyperpara(scaled_train, labels_train, max_trees=100, show_fig=False):

    n_trees = list(range(1,max_trees,5))
    all_train = []
    all_val = []
    X2 = scaled_train
    y2 = labels_train
    #    Repeat the experiment 20 times, use 20 random splits in which class balance is retained
    sss = model_selection.StratifiedShuffleSplit(n_splits=20, test_size=0.5, random_state=None)
    #print(sss)
    for train_index, val_index in sss.split(X2, y2):
        train_scores = []
        val_scores = []
        
        split_X_train = X2[train_index]
        split_y_train = y2[train_index]
        split_X_val = X2[val_index]
        split_y_val = y2[val_index]

        for n in n_trees:
            clf_RF = RandomForestClassifier(n_estimators=n)
            #print(clf_RF)
            #clf_knn = neighbors.KNeighborsClassifier(n_neighbors=k)
            
            clf_RF.fit(split_X_train, split_y_train)

            # Test the classifier on the training data and plot
            score_train = clf_RF.score(split_X_train, split_y_train)
            score_val = clf_RF.score(split_X_val, split_y_val)

            train_scores.append(score_train)
            val_scores.append(score_val)
            
        all_train.append(train_scores)
        all_val.append(val_scores)
        

    # Create numpy array of scores and calculate the mean and std
    all_train = np.array(all_train)
    all_val = np.array(all_val)

    train_scores_mean = all_train.mean(axis=0)
    train_scores_std = all_train.std(axis=0)

    val_scores_mean = all_val.mean(axis=0)
    val_scores_std = all_val.std(axis=0)

    if show_fig:
        # Plot the mean scores and the std as shading
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.grid()
        ax.fill_between(n_trees, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
        ax.fill_between(n_trees, val_scores_mean - val_scores_std,
                            val_scores_mean + val_scores_std, alpha=0.1,
                            color="g")
        ax.plot(n_trees, train_scores_mean, 'o-', color="r",
                label="Training score")
        ax.plot(n_trees, val_scores_mean, 'o-', color="g",
                label="validation score")
        plt.show()
    return (train_scores_mean, train_scores_std, val_scores_mean, val_scores_std, plt)


def random_forest_algoritm(train_data, train_labels, test_data, test_labels, n_estimators=40):
    
    clf = RandomForestClassifier(n_estimators)
    clf.fit(train_data, train_labels)
    y_pred = clf.predict(test_data)
    f1, prec, acc, recall = scoring(test_labels, y_pred)
    #test_score=clf.score(test_data,test_labels)
    return f1, prec, acc, recall