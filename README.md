# CS420-Machine-Learning-Project

## Contents

* [Summary](#summary)
* [Prerequisite](#prerequisite)
* [Traditional Methods](#traditional-methods)
* [CNN](#cnn)
* [Capsule](#capsule)
* [Result](#result)
* [Structure](#structure)
* [Team Member](#team-member)

## Summary

## Prerequisite

* Python
* Pytorch==0.4.0
* Numpy==1.14.2
* Scikit Learn==0.19.1
* tqdm==4.11.2

## Traditional Methods

## CNN

We try three kinds of network architectures, SimpleCNN, VGG and Resnet.

Usage:

```
	cd deep-learning-method
	python tools/train.py [--gpu GPU-ID] [--dataset DATASET] [--net NET] 
	[--resume path/to/resume/from] [--logdir path/to/log/dir]

	arguments:
	--gpu GPU-ID    		the id of GPU card to use
	--dataset DATASET  		the dataset to train and test
	--net NET     			the network structure to use
					options: ConvNet,vgg16,vgg11,res18,res34,res50,res101      
	--resume path/to/resume/from	resume training from the given path
	--logdir path/to/log/dir	directory to log 
```

## Capsule

Usage:

```
	cd capsule
	CUDA_VISIBLE_DEVICES=0 python3 capsule_network.py
```

## Result

## Structure

## Team Member

* [Weichao Mao](https://github.com/xizeroplus)
* [Ruiheng Chang](https://github.com/crh19970307)



