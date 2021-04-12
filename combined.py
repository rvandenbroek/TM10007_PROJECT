import math
import pandas as pd
import numpy as np 
from brats.load_data import load_data
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from PCA import PCA_algorithm
from sklearn.svm import SVC
from k_fold_cross_val import k_fold_cross_validation
from KNN import knn_classifier
# from SVM import SVM_algorithm
# from SVM import SVM_hyper
# from SVM import SVM_PCA
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
import seaborn
import matplotlib.pyplot as plt
from statistics import mean
from statistics import stdev
from score import scoring

# Definitions preprocessing
def dropnan(data, threshold):
    '''First of all, cetrain strings and values in the dataset are turned into NaN.
    With this function, features with more than a certain amount of NaN values will be
    excluded from the dataset. The inputs are the contaminated dataset and the
    threshold value. The output is the cleaned dataset. '''
    # replace '#DIV/0!' and infinity values with NaN
    data = data.replace(r'#DIV/0!', np.nan, regex=True)
    data = data.replace([np.inf, -np.inf], np.nan, regex=True)
    # drop the features with more than threshold NaN's
    data = data.dropna(thresh=threshold, axis=1)
    return data


def imputation(train_data, test_data):
    '''In order to substitute the NaN values rather then delete them, a kNN imputer function is
    used to impute the missing data. This function is based on the train set and subsequently
    applied on the test set. This ensures the model is completely trained on the train set rather than
    the test set. The inputs are the trainset and the testset, the outputs are the same sets with imputed 
    values.
    '''
    # impute the still existing NaN's 
    imputer = KNNImputer(n_neighbors=3, weights="uniform")
    imputed_train = imputer.fit_transform(train_data)
    imputed_test = imputer.transform(test_data)

    return imputed_train, imputed_test


def robust_scaler(train_data, test_data):
    """This scaler removes the median and scales the data according to the quantile range.
    The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).
    The scaler is fit on the train dataset and applied to the test data set to prevent training on the test set. 
    """
    scaler = preprocessing.RobustScaler()
    scaler.fit(train_data)
    scaled_train = scaler.transform(train_data)
    scaled_test = scaler.transform(test_data)

    return (scaled_train, scaled_test)


def build_model_and_results(cv_10fold, classifier, parameters, show_fig=False):
    '''This function splits the data and finds the best hyperparameter per split.
    This hyperparameter is applied in a classification on the test set in 10 folds. The accuracy,
    F1 score, precision, recall, confusion matrices, true positives and aucs are returned'''
    accuracies = []
    f1_score = []
    precision = []
    recall_score = []
    conf_matrix_list = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()

    # Loop over the folds
    for i, (validation_index, test_index) in enumerate(cv_10fold.split(data, labels)):
        
        # Split the data
        x_validation = data.iloc[validation_index]
        y_validation = labels[validation_index]

        x_test = data.iloc[test_index]
        y_test = labels[test_index]

        # Preprocessing of data
        imputed_train, imputed_test = imputation(x_validation, x_test)
        scaled_train, scaled_test = robust_scaler(imputed_train, imputed_test)
        pca_train, pca_test = PCA_algorithm(scaled_train, scaled_test)

        # Create a grid search to find the optimal k using a gridsearch and 10-fold cross validation
        grid_search = model_selection.GridSearchCV(classifier, parameters, cv=cv_10fold, scoring='roc_auc')
        grid_search.fit(pca_train, y_validation)

        # Get resulting classifier with best hyperparameter
        clf = grid_search.best_estimator_

        # Test the classifier on the test data
        predicted = clf.predict(pca_test)

        # Scores per fold
        f1, prec, acc, recall = scoring(y_test, predicted)
        f1_score.append(f1)
        accuracies.append(acc)
        precision.append(prec)
        recall_score.append(recall)

        # Confusion matrix per fold
        conf_matrix = metrics.confusion_matrix(y_test, predicted)
        conf_matrix_list.append(conf_matrix)

        # plot ROC curve per fold
        viz = metrics.plot_roc_curve(clf, pca_test, y_test,
                                    name='ROC fold {}'.format(i),
                                    alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    # Combine scores of the folds into a dataframe with mean and std accuracy, F1 score, precision and recall
    scoring_df = {'mean accuracy': mean(accuracies), 'std accuracy': stdev(accuracies),
                  'mean f1': mean(f1_score), 'std f1': stdev(f1_score),
                  'mean precision': mean(precision), 'std accuracy': stdev(precision),
                  'mean recall': mean(recall_score), 'std recall': stdev(recall_score)}
    print(scoring_df)
    
    # Plot combined ROC curves
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating curve")
    ax.legend(loc="lower right")
    plt.show()

    # Combine results of confusion matrices into one mean confusion matrix
    mean_of_conf_matrix = np.mean(conf_matrix_list, axis=0)
    conf_matrix_df = pd.DataFrame.from_dict({
            'Actual: GBM': mean_of_conf_matrix[0],
            'Actual: LGG': mean_of_conf_matrix[1]},
             orient='index', columns=['Predicted: GBM', 'Predicted: LGG'])
    seaborn.heatmap(conf_matrix_df/np.sum(np.sum(conf_matrix_df)), annot= True, fmt='.2%', cmap='Blues')
    plt.show()
    return

              
data = load_data()
threshold = math.floor(len(data)*0.8)
data = dropnan(data, threshold)
labels = np.array(data['label'])
data.pop('label')

cv_10fold = model_selection.StratifiedKFold(n_splits=10)
parameters_knn = {"n_neighbors": list(range(1, 30, 2))}
knn = neighbors.KNeighborsClassifier()
parameters_RF = {"n_estimators": list(range(1, 51, 5))}
RF = RandomForestClassifier()
parameters_svm = {"C": [0.4, 0.6, 0.8, 1, 1.2, 1.4], "coef0": list(range(1, 25, 5))}
svm = SVC(probability=True, gamma='scale', kernel='poly')

# build_model_and_results(cv_10fold, knn, parameters_knn)
# build_model_and_results(cv_10fold, svm, parameters_svm)
build_model_and_results(cv_10fold, RF, parameters_RF)
