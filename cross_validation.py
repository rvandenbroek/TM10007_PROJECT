
from sklearn import model_selection
from sklearn import svm
from sklearn import feature_selection
import matplotlib.pyplot as plt

# Revoming features with low variance: VarianceThreshold 
# Recursive feature elimination: RFE or RFECV? 
# Feature selection using SelectfromModel: based on threshold of importance weights 
# Sequential feature selector: Sequential cross validation based feature selection does not rely on importance weights. 
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html



def rfe(X2,y2):
    # Create the RFE object and compute a cross-validated score.
    svc = svm.SVC(kernel="linear")

    # classifications
    rfecv = feature_selection.RFECV(
        estimator=svc, step=1, 
        cv=model_selection.StratifiedKFold(4),
        scoring='roc_auc')
    rfecv.fit(X2, y2)

# Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()