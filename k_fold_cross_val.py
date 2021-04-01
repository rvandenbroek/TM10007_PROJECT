

def k_fold_cross_validation(traindata, traindata_labels, n_splits_hyper_para =10):
    """This k_fold cross validation allows you to split your training data
    in a validation set and a train set. Required imputs: train_data, train_data labels,
    and optionally you can select the number of splits for the cross validation. 
    Output: X_train= the training data, X_val = validation data, y_train = training
    labels, y_val= validation labels.  """
    X = np.array(traindata)
    y = np.array(traindata_labels)

    rkf = RepeatedKFold(n_splits=n_splits_hyper_para, n_repeats=n_splits_hyper_para)

    for train_index, test_index in rkf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index] 
        return X_train, X_val, y_train, y_val # even kijken waar deze return moet komen
