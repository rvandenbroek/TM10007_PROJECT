from sklearn.svm import SVC
from sklearn import model_selection
import numpy as np 
import matplotlib.pyplot as plt

def SVM_algorithm(X_train, Y_train, X_test, Y_test):
    clf = SVC(kernel='linear', degree=3, gamma='scale')
    clf.fit(X_train, Y_train)
    score_test = clf.score(X_test, Y_test)
    #score_test = clf.predict(X_test)

    return score_test



    # kernel = 'linear', 'rbf', 'poly'
    # coef0 = integer
    # degree = integer
    # C (slack, Regularization parameter) = integer (positive) 

    # wat ze ook doen is 
    # clf.fit(X_all, Y_all) 
    # clf.predict(X_all) predicten met alleen X 
    # Dan die uitkomst vergelijken met Y? 



#hyperparameters: degree of kernel, homogeneity coef0. Slack of parameter
def SVM_hyper (X_train_pca, y_train, X_test_pca, y_test):
    k_list = list(range(1, 50, 1))
    all_train = []
    all_test = []
    
    sss = model_selection.StratifiedShuffleSplit(n_splits=20, test_size=0.5, random_state=0)
    for train_index, test_index in sss.split(X_train_pca, y_train):
        train_scores = []
        val_scores = []
    
        split_X_train = X_train_pca[train_index]
        split_y_train = y_train[train_index]
        split_X_val = X_train_pca[test_index]
        split_y_val = y_train[test_index]

        for k in k_list:
            clf_SVM = SVC(kernel='poly', degree=3, gamma='scale', coef0=k, C=1)

            #clf_knn = neighbors.KNeighborsClassifier(n_neighbors=k)
            clf_SVM.fit(split_X_train, split_y_train)

        # Test the classifier on the training data and plot
            score_train = clf_SVM.score(split_X_train, split_y_train)
            score_val = clf_SVM.score(split_X_val, split_y_val)

            train_scores.append(score_train)
            val_scores.append(score_val)
        
        all_train.append(train_scores)
        all_test.append(val_scores)

    # Create numpy array of scores and calculate the mean and std
    all_train = np.array(all_train)
    all_test = np.array(all_test)

    train_scores_mean = all_train.mean(axis=0)
    train_scores_std = all_train.std(axis=0)

    test_scores_mean = all_test.mean(axis=0)
    test_scores_std = all_test.std(axis=0)

    # Plot the mean scores and the std as shading
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.fill_between(k_list, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax.fill_between(k_list, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    ax.plot(k_list, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(k_list, test_scores_mean, 'o-', color="g",
            label="Test score")
    plt.show()



from sklearn import decomposition 


def SVM_PCA(X_train_scaled, y_train, X_test_scaled, y_test):
    k_list = list(range(1, 10, 1))
    all_train = []
    all_test = []

    for k in k_list:
        pca = decomposition.PCA(n_components=k)
        pca.fit(X_train_scaled)
        X_train_pca = pca.transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)


    
        sss = model_selection.StratifiedShuffleSplit(n_splits=20, test_size=0.5, random_state=0)
        for train_index, test_index in sss.split(X_train_pca, y_train):
            train_scores = []
            val_scores = []
    
            split_X_train = X_train_pca[train_index]
            split_y_train = y_train[train_index]
            split_X_val = X_train_pca[test_index]
            split_y_val = y_train[test_index]

         
            clf_SVM = SVC(kernel='poly', degree=3, gamma='scale', coef0=25, C=1)

            #clf_knn = neighbors.KNeighborsClassifier(n_neighbors=k)
            clf_SVM.fit(split_X_train, split_y_train)

            # Test the classifier on the training data and plot
            score_train = clf_SVM.score(split_X_train, split_y_train)
            score_val = clf_SVM.score(split_X_val, split_y_val)

            train_scores.append(score_train)
            val_scores.append(score_val)
        
            all_train.append(train_scores)
            all_test.append(val_scores)

    # Create numpy array of scores and calculate the mean and std
        all_train = np.array(all_train)
        all_test = np.array(all_test)

        train_scores_mean = all_train.mean(axis=0)
        train_scores_std = all_train.std(axis=0)

        test_scores_mean = all_test.mean(axis=0)
        test_scores_std = all_test.std(axis=0)

    # Plot the mean scores and the std as shading
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.fill_between(k_list, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax.fill_between(k_list, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    ax.plot(k_list, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(k_list, test_scores_mean, 'o-', color="g",
            label="Test score")
    plt.show()
