import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


n_samples = 60000
fig_w = 45

data = np.fromfile("mnist_train\\mnist_train_data",dtype=np.uint8)
X_train = data.reshape(n_samples, -1)[: 2000]
Y_train = np.fromfile("mnist_train\\mnist_train_label",dtype=np.uint8)[: 2000]

X_test = np.fromfile("mnist_test\\mnist_test_data",dtype=np.uint8).reshape(10000, -1)[: 500]
Y_test = np.fromfile("mnist_test\\mnist_test_label" ,dtype=np.uint8)[: 500]

for pca_ratio in np.arange(0.4, 1.01, 0.1):
    for C in (10 ** i for i in range(-4, 2, 1)):
        for gamma in (10 ** j for j in range(-5, 1, 1)):

            file = open('SVM_output.txt', 'a')
            file.write('pca_ratio = ' + str(pca_ratio) + '  C = ' + str(C) + '  gamma = ' + str(gamma))
            print('pca_ratio = ' + str(pca_ratio) + '  C = ' + str(C) + '  gamma = ' + str(gamma))
            
            pca = PCA(n_components=pca_ratio)
            pca.fit(X_train)
            X_train_pca = pca.transform(X_train)
            print(X_train_pca.shape)
            print(pca.n_components_)
            X_test_pca = pca.transform(X_test)

            clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)

            clf.fit(X_train_pca, Y_train)

            Y_pred = clf.predict(X_test_pca)
            #file.write(metrics.classification_report(Y_test, Y_pred))
            #file.write('\n')
            file.write(' Accuracy = ' + str(metrics.accuracy_score(Y_test, Y_pred)))
            file.write('\n')
            print('Accuracy = ' + str(metrics.accuracy_score(Y_test, Y_pred)))
            file.close()