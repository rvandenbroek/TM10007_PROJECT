from sklearn.svm import SVC
from sklearn import neighbors
# Fit kNN


def knn_classifier(X_train_pca, y_train, X_test_pca, y_test):

    knn = neighbors.KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train_pca, y_train)
    score_train = knn.score(X_train_pca, y_train)
    score_test = knn.score(X_test_pca, y_test)
    return score_train score_test


def SVM(X_train, Y_train, X_test, Y_test):
    clf = SVC(kernel='poly', degree=3, gamma='scale')
    clf.fit(X_train, Y_train)
    clf.score(X_test, Y_test)

    # wat ze ook doen is 
    # clf.fit(X_all, Y_all) 
    # clf.predict(X_all) predicten met alleen X 
    # Dan die uitkomst vergelijken met Y? 



#hyperparameters: degree of kernel, homogeneity coef0. Slack of parameter


