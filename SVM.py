import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


n_samples = 60000
fig_w = 45

pca_ratio = 0.65
C = 6
gamma = 6e-7

data = np.fromfile("mnist_train/new_train_data",dtype=np.uint8)
X_train = data.reshape(n_samples, -1)
Y_train = np.fromfile("mnist_train/mnist_train_label",dtype=np.uint8)

X_test = np.fromfile("mnist_test/new_test_data",dtype=np.uint8).reshape(10000, -1)
Y_test = np.fromfile("mnist_test/mnist_test_label" ,dtype=np.uint8)

for pca_ratio in np.arange(0.95, 0.96, 0.05):
	for C in [5, 10]:
		for gamma in [2e-7, 5e-7, 1e-6]:

			file = open('SVMwith_output2.txt', 'a')
			file.write('pca_ratio = ' + str(pca_ratio) + '  C = ' + str(C) + '  gamma = ' + str(gamma))
			print('pca_ratio = ' + str(pca_ratio) + '  C = ' + str(C) + '  gamma = ' + str(gamma))

			pca = PCA(n_components=pca_ratio)
			pca.fit(X_train)
			X_train_pca = pca.transform(X_train)
			X_test_pca = pca.transform(X_test)

			clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)

			clf.fit(X_train_pca, Y_train)

			Y_pred = clf.predict(X_test_pca)
			file.write('Accuracy = ' + str(metrics.accuracy_score(Y_test, Y_pred)))
			file.write('\n')
			# file.write(metrics.classification_report(Y_test, Y_pred))
			file.write('\n')
			print('Accuracy = ' + str(metrics.accuracy_score(Y_test, Y_pred)))
			file.close()