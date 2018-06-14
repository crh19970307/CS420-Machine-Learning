import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.decomposition import PCA


n_samples = 60000
fig_w = 45

pca_ratio = 0.6
neighbors = 5

data = np.fromfile("../mnist_train/mnist_train_data",dtype=np.uint8)
X_train = data.reshape(n_samples, -1)
Y_train = np.fromfile("../mnist_train/mnist_train_label",dtype=np.uint8)
X_test = np.fromfile("../mnist_test/mnist_test_data",dtype=np.uint8).reshape(10000, -1)
Y_test = np.fromfile("../mnist_test/mnist_test_label" ,dtype=np.uint8)


pca = PCA(n_components=pca_ratio)
X_train_pca = pca.fit(X_train).transform(X_train)


clf = KNeighborsClassifier(n_neighbors=neighbors)
clf.fit(X_train_pca, Y_train)

X_test_pca = pca.transform(X_test)
Y_pred = clf.predict(X_test_pca)

file = open('KNNwithout_result.txt', 'a')
file.write('pca_ratio = ' + str(pca_ratio) + ' n_components = ' + str(neighbors) + ' accuracy = ')
file.write(str(metrics.accuracy_score(Y_test, Y_pred)) + '\n')
file.write(metrics.classification_report(Y_test, Y_pred))
file.close()

print('pca_ratio = ' + str(pca_ratio) + ' n_components = ' + str(neighbors))
print('Accuracy = ' + str(metrics.accuracy_score(Y_test, Y_pred)) + '\n')
