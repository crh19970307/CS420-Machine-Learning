"""
This file implements the KNN classifier, and fits KNN on the unprocessed dataset.
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.decomposition import PCA


n_samples = 60000
fig_w = 45

# set parameter values
pca_ratio = 0.6
neighbors = 5

# load dataset from files
data = np.fromfile("../mnist_train/mnist_train_data",dtype=np.uint8)
X_train = data.reshape(n_samples, -1)
Y_train = np.fromfile("../mnist_train/mnist_train_label",dtype=np.uint8)
X_test = np.fromfile("../mnist_test/mnist_test_data",dtype=np.uint8).reshape(10000, -1)
Y_test = np.fromfile("../mnist_test/mnist_test_label" ,dtype=np.uint8)

# perform PCA on training data
pca = PCA(n_components=pca_ratio)
X_train_pca = pca.fit(X_train).transform(X_train)

# fit the KNN classifier on training data
clf = KNeighborsClassifier(n_neighbors=neighbors)
clf.fit(X_train_pca, Y_train)

# predict on the testing data
X_test_pca = pca.transform(X_test)
Y_pred = clf.predict(X_test_pca)

# output the results to file
file = open('KNNwithout_result.txt', 'a')
file.write('pca_ratio = ' + str(pca_ratio) + ' n_components = ' + str(neighbors) + ' accuracy = ')
file.write(str(metrics.accuracy_score(Y_test, Y_pred)) + '\n')
file.write(metrics.classification_report(Y_test, Y_pred))
file.close()

print('pca_ratio = ' + str(pca_ratio) + ' n_components = ' + str(neighbors))
print('Accuracy = ' + str(metrics.accuracy_score(Y_test, Y_pred)) + '\n')
