import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

class vgg(nn.Module):
    def __init__(self,layer,num_classes):
        super(vgg,self).__init__()
        self.cfg={
            'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
                  512, 512, 512, 512, 'M'],
                }
        self.num_classes=num_classes
        self.features=self.make_layers(self.cfg[layer])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Linear(512,self.num_classes)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,(2./n)**0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self,x):
        out=self.features(x)
        out=out.view(out.size(0),-1)
        out=self.classifier(out)
        return out
    
    def make_layers(self,layer_list):
        layers=[]
        in_channels=1
        for layer in layer_list:
            if layer=='M':
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            else :
                conv2d = nn.Conv2d(in_channels, layer, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(layer), nn.ReLU(inplace=True)]
                in_channels = layer
        return nn.Sequential(*layers)

def vgg16(num_classes):
    return vgg('vgg16',num_classes)
            
def vgg11(num_classes):
    return vgg('vgg11',num_classes)
            
    