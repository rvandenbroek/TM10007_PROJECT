from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from main import scaled_data
from main import labels_train
from main import data_test
from preprocessing import imputation
from preprocessing import robust_scaler

n_trees = [1, 5, 10, 50, 100, 500]
for n_tree in n_trees:
    clf = RandomForestClassifier(n_estimators=n_tree)

    clf.fit(scaled_data, labels_train)
    #clf.fit(scaled_data, labels_train)
    imp= imputation(data_test)
    rob=robust_scaler(imp)
    pred=clf.predict(rob)
    print(f'number of trees = {n_tree}, pred= {pred}')
    #print((pred==data_test))
#print(type(pred))

#print(clf.predict([scaled_data]))