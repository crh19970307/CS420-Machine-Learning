import _init_paths
import torch 
import torch.nn as nn
import numpy as np
import torchvision
import os
import argparse

from utils.logger import Logger
from datasets.mnist import My_MNIST
from nets.SimpleCNN import ConvNet
from nets.vgg import vgg16,vgg11
from nets.resnet import res18,res34,res50,res101
#from nets.resnet import res18

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Parse arguments template')
    parser.add_argument('--gpu', dest='gpu', help='GPU id to use',default='0')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset',default='mnist')
    parser.add_argument('--net', dest='net', help='Network to use')
    parser.add_argument('--resume', dest='resume', help='path/to/resume/from')
    parser.add_argument('--logdir', dest='logdir', help='path/to/logdir')
    # Example: parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',choices=NETS.keys(), default='res101')
    args = parser.parse_args()

    return args

if __name__=='__main__':

    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    num_epochs = 100
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001

    train_dataset = My_MNIST(root='data/mnist',train=True )
    test_dataset = My_MNIST(root='data/mnist',train=False)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              shuffle=False)
    
    if args.net=='SimpleCNN':
        model = ConvNet(num_classes).to(device)
    elif args.net=='vgg16':
        model = vgg16(num_classes).to(device)
    elif args.net=='vgg11':
        model = vgg11(num_classes).to(device)    
    elif args.net=='res18':
        model = res18(num_classes).to(device)
    elif args.net=='res34':
        model = res34(num_classes).to(device)
    elif args.net=='res50':
        model = res50(num_classes).to(device)
    elif args.net=='res101':
        model = res101(num_classes).to(device)     
        
    if args.resume :
        model.load_state_dict(torch.load(args.resume))
    
    if args.logdir:
        log_dir=args.logdir
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir) 
    else:
        log_dir=os.path.join('tensorboard',args.dataset,args.net)        
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)    
            
    logger = Logger(log_dir)
        
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        output_dir=os.path.join('output',args.dataset,args.net)        
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)     
        if epoch == num_epochs:
            torch.save(model.state_dict(), output_dir+'/'+args.net+'-%04d'%(epoch)+'.ckpt')
        # Test the model
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy=100.0 * correct / total
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(accuracy))
        info = { 'loss': loss.item(), 'accuracy': accuracy }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch+1)
        
            
            

