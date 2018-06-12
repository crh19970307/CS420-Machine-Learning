import torch 
import torch.nn as nn
import numpy as np
import torchvision
import os


class My_MNIST(torch.utils.data.Dataset):
    def __init__(self,root,train):
        if train:
            self.data=np.fromfile(root+'/mnist_train/'+"mnist_train_data",dtype=np.uint8)
            self.label = np.fromfile(root+'/mnist_train/'+"mnist_train_label",dtype=np.uint8)
        else:
            self.data=np.fromfile(root+'/mnist_test/'+"mnist_test_data",dtype=np.uint8)
            self.label = np.fromfile(root+'/mnist_test/'+"mnist_test_label",dtype=np.uint8)
        self.data=self.data.reshape(self.label.shape[0],45,45)
    def __len__(self):
        return self.label.shape[0]
    def __getitem__(self,idx):
        return torch.from_numpy(np.array([self.data[idx]])).float(),torch.tensor(int(self.label[idx]))