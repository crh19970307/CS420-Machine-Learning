import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn import svm


n_samples = 60000
fig_w = 45
neighbors = 1
pca_ratio = 0.8
gamma = 1e-6
C = 6

data = np.fromfile("mnist_train/new_train_data",dtype=np.uint8)
X_train = data.reshape(n_samples, -1)
Y_train = np.fromfile("mnist_train/mnist_train_label",dtype=np.uint8)

X_test = np.fromfile("mnist_test/new_test_data",dtype=np.uint8).reshape(10000, -1)
Y_test = np.fromfile("mnist_test/mnist_test_label" ,dtype=np.uint8)

pca = PCA(n_components=pca_ratio)
X_train_pca = pca.fit(X_train).transform(X_train)
X_test_pca = pca.transform(X_test)

clf1 = KNeighborsClassifier(n_neighbors=neighbors)
clf2 = svm.SVC(kernel='rbf', C=C, gamma=gamma, probability=True)
eclf = VotingClassifier(estimators=[('knn', clf1), ('svm', clf2)], voting='soft')

eclf = eclf.fit(X_train_pca, Y_train)

Y_pred = eclf.predict(X_test_pca)

file = open('voting_final_result.txt', 'a')
file.write('pca_ratio = ' + str(pca_ratio) + ' gamma = ' + str(gamma) + ' accuracy = ')
file.write(str(metrics.accuracy_score(Y_test, Y_pred)) + '\n')
file.write(metrics.classification_report(Y_test, Y_pred))
file.close()

print('pca_ratio = ' + str(pca_ratio) + ' gamma = ' +str(gamma) + '\n')
print('Accuracy = ' + str(metrics.accuracy_score(Y_test, Y_pred)) + '\n')