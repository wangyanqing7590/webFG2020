#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torchvision
from torch.utils.data import DataLoader
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from Imagefolder_modified import Imagefolder_modified
from resnet import ResNet50
from PIL import ImageFile # Python：IOError: image file is truncated 的解决办法
ImageFile.LOAD_TRUNCATED_IMAGES = True

import time

epochs = 30
data_set = 5000
batch_size = 64

class Manager(object):
    def __init__(self):
        """
        Prepare the network, criterion, Optimizer and data
        Arguments:
            options [dict]  Hyperparameter
        """
        

        net = ResNet50(data_set)
        print('Loading model from resnet50_uniform_e200.pth')
        
        checkpoint = torch.load('/data1/wangyanqing/models/webFG2020/resnet50_uniform_e200.pth', map_location=torch.device('cpu'))          
        model_state = checkpoint['state_dict_best']
        
        # self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None
        new_weights = {}
        weights = model_state['feat_model']
        for k, v in weights.items():
            if(k[:12] == 'module.conv1'):
                k = 'features.0' + k[12:]
            elif(k[:10] == 'module.bn1'):
                k = 'features.1' + k[10:]
            elif(k[:13] == 'module.layer1'):
                k = 'features.4' + k[13:]
            elif(k[:13] == 'module.layer2'):
                k = 'features.5' + k[13:]
            elif(k[:13] == 'module.layer3'):
                k = 'features.6' + k[13:]
            elif(k[:13] == 'module.layer4'):
                k = 'features.7' + k[13:]
            else:
                print('match error : {}'.format(k))
            new_weights[k] = v

        # weights = {k: weights[k] for k in weights if k in net.state_dict()}
        x = net.state_dict()

        x.update(new_weights)

        net.load_state_dict(x) 

        
        for params in net.features.parameters():
            params.required_grad = False
        print("useing freeze......................")

        if torch.cuda.device_count() >= 1:
            self._net = torch.nn.DataParallel(net).cuda()
            print('cuda device : ', torch.cuda.device_count())
        else:
            raise EnvironmentError('This is designed to run on GPU but no GPU is found')
        




        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        
        self._optimizer = torch.optim.SGD(self._net.module.fc.parameters(), lr=0.00314159, momentum=0.9)
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=2, gamma=0.9)

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=256),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        data_dir = '/data1/wangyanqing/projects/webFG2020/data_%d' %(data_set)
        train_data = Imagefolder_modified(os.path.join(data_dir, 'train.txt'), transform=train_transform)
        test_data = Imagefolder_modified(os.path.join(data_dir, 'val.txt'), transform=test_transform, cached=False)

        self._train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=4, pin_memory=True)
        self._test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=False, num_workers=4, pin_memory=True)



    def train(self):
        """
        Train the network
        """
        print('Training ... ')
        best_accuracy = 0.0
        best_epoch = 0
        print('Epoch\tTrain Loss\tTrain Accuracy\tTest Accuracy\tEpoch Runtime\tlr')
        for t in range(epochs):

            epoch_start = time.time()
            epoch_loss = []
            num_correct = 0
            num_total = 0
            num_remember = 0
            steps = 0
            for X, y, id, path in self._train_loader:
                # Enable training mode
                self._net.train(True)
                # Data
                X = X.cuda()
                y = y.cuda()
                # Forward pass
                score = self._net(X)  # score is in shape (N, 200)

                loss = self._criterion(score, y)

                epoch_loss.append(loss.item())
                # Prediction
                closest_dis, prediction = torch.max(score.data, 1)

                num_total += y.size(0)  # y.size(0) is the batch size
                num_correct += torch.sum(prediction == y.data).item()

                # Clear the existing gradients
                self._optimizer.zero_grad()
                # Backward
                loss.backward()
                self._optimizer.step()
                steps += 1
                if(steps % 370 == 0):
                    print('%d\t%4.2f%%\t\t%4.3f\t\t%4.2f%%\t\t%s' % (t + 1,steps / 370,  loss.item(),torch.sum(prediction == y.data).item() / y.size(0),time.asctime(time.localtime())))
            # Record the train accuracy of each epoch
            train_accuracy = 100 * num_correct / num_total
            test_accuracy = self.test(self._test_loader)
            self._scheduler.step()

            epoch_end = time.time()
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = t +1
                print('*', end='')
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f\t\t%6f\t\t' % (t + 1, sum(epoch_loss) / len(epoch_loss),
                                                            train_accuracy, test_accuracy,
                                                            epoch_end - epoch_start, self._optimizer.param_groups[0]["lr"] ))
            


        print('******\n'
        'Best Accuracy 1: [{0:6.2f}%], at Epoch [{1:03d}] '
        '\n******'.format(best_accuracy, best_epoch))

        print('-----------------------------------------------------------------')

    def test(self, dataloader):
        """
        Compute the test accuracy

        Argument:
            dataloader  Test dataloader
        Return:
            Test accuracy in percentage
        """
        self._net.train(False) # set the mode to evaluation phase
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for X, y,_,_ in dataloader:
                # Data
                X = X.cuda()
                y = y.cuda()
                # Prediction
                score = self._net(X)
                _, prediction = torch.max(score, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data).item()
        self._net.train(True)  # set the mode to training phase
        return 100 * num_correct / num_total

if __name__ == '__main__':
    manager = Manager()
    manager.train()