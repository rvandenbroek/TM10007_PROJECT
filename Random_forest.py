
#allemaal gekopieerd en aangepast van de Colab maar werkt nog niet!

from main import scaled_data
from main import labels_train
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#n_trees = [1, 5, 10, 50, 100]
n_trees = [5]
num =1
    
# Now use the classifiers on all datasets
fig = plt.figure(figsize=(24,8*1))

clf = RandomForestClassifier(n_estimators=5)
clf.fit(scaled_data, labels_train)
ax = fig.add_subplot(7, 3, num + 1)
ax.scatter(scaled_data[:, 0], scaled_data[:, 1], marker='o', c=labels_train,
    s=25, edgecolor='k', cmap=plt.cm.Paired)
colorplot(clf, ax, scaled_data[:, 0], scaled_data[:, 1])
y_pred = clf.predict(scaled_data)
t = ("Misclassified: %d / %d" % ((labels_train != y_pred).sum(), scaled_data.shape[0]))
ax.set_title(t)
num += 1