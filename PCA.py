from sklearn import decomposition 

def PCA(X_train_scaled):
    pca = decomposition.PCA(n_components=2)
    pca.fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    #X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca

# Dimensionality reduction
# Class visualization 