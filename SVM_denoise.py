import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os

# im = Image.fromarray(data[ind])
# arr = np.asarray(im)

def save_image_as_txt(data, filename):
    np.savetxt(filename, data, fmt="%i")


n_samples = 60000
fig_w = 45

data = np.fromfile("mnist_train\\mnist_train_data",dtype=np.uint8)
X_train = data.reshape(n_samples, -1)[: 1000]
data = data.reshape(n_samples, fig_w, fig_w)

Y_train = np.fromfile("mnist_train\\mnist_train_label",dtype=np.uint8)[: 1000]

X_test = np.fromfile("mnist_test\\mnist_test_data",dtype=np.uint8).reshape(10000, -1)[: 500]
Y_test = np.fromfile("mnist_test\\mnist_test_label" ,dtype=np.uint8)[: 500]









pca = PCA(n_components=0.85, whiten=True)
new_X_train = pca.fit_transform(X_train)
print(pca.n_components_)
new_X_test = pca.transform(X_test)

clf = svm.SVC(kernel='linear')
clf.fit(new_X_train, Y_train)

Y_pred = clf.predict(new_X_test)
print(metrics.classification_report(Y_test, Y_pred))
print(metrics.accuracy_score(Y_test, Y_pred))