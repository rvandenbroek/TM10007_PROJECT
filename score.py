from sklearn.metrics import f1_score, average_precision_score, accuracy_score, recall_score, confusion_matrix

def scoring(y_true_label, y_prediction):

    #y_pred = knn.predict(X_test_pca)
    y_prediction = [1 if i=='GBM' else 0 for i in y_prediction]
    y_true_label = [1 if i=='GBM' else 0 for i in y_true_label]

    f1 = f1_score(y_true_label, y_prediction)
    prec = average_precision_score(y_true_label, y_prediction)
    acc = accuracy_score(y_true_label, y_prediction)
    recall = recall_score(y_true_label, y_prediction)
    CM = confusion_matrix(y_true_label, y_prediction)
    print(CM)
    return(f1, prec, acc, recall)
