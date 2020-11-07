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
from resnet import ResNet18_Normalized, ResNet50
from PIL import ImageFile # Python：IOError: image file is truncated 的解决办法
ImageFile.LOAD_TRUNCATED_IMAGES = True

import time



class Manager(object):
    def __init__(self, options):
        """
        Prepare the network, criterion, Optimizer and data
        Arguments:
            options [dict]  Hyperparameter
        """
        print('------------------------------------------------------------------------------')
        print('Preparing the network and data ... ')
        self._options = options
        self._path = options['path']
        os.popen('mkdir -p ' + self._path)
        self._data_base = options['data_base']
        self._class = options['n_classes']
        self._denoise = options['denoise']
        self._drop_rate = options['drop_rate']
        self._smooth = options['smooth']
        self._label_weight = options['label_weight']
        self._tk = options['tk']
        self._warmup= options['warmup']
        self._step = options['step']
        self._prt_model = options['prt_model']
        print('Basic information: ','net:',options['net'], 'prt_model:',self._prt_model,'data:',self._data_base,'  lr:', self._options['base_lr'],'  w_decay:', self._options['weight_decay'])
        print('Parameter information: ','denoise:',self._denoise,'  drop_rate:',self._drop_rate,'  smooth:',self._smooth,'  label_weight:',self._label_weight,'  tk:',self._tk, '  warmup:',self._warmup)
        print('------------------------------------------------------------------------------')
        # Network
        # We recommend resnet18, which takes less time to train
        if options['net'] == 'resnet152':
            print("to be code...")
        elif options['net'] == 'resnet50':
            NET = ResNet50
        else:
            raise AssertionError('Not implemented yet')

        net = NET(n_classes=options['n_classes'])   


        if torch.cuda.device_count() >= 1:
            self._net = torch.nn.DataParallel(net).cuda()
            print('cuda device : ', torch.cuda.device_count())
        else:
            raise EnvironmentError('This is designed to run on GPU but no GPU is found')
        
 
        # print(self._net)
        # Criterion
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Optimizer


        if self._step == 1:

            print('Loading model from {}_uniform_e200.pth'.format(options['net']))
            
            checkpoint = torch.load('/data1/wangyanqing/models/webFG2020/{}_uniform_e200.pth'.format(options['net']))          
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
            x = net.state_dict()
            x.update(new_weights)
            net.load_state_dict(x) 
            for params in net.features.parameters():
                params.required_grad = False
            params_to_optimize = self._net.module.fc.parameters()
            print('step1')
        else:
            self._net.load_state_dict(torch.load(os.path.join(self._path, '{}_step1.pth'.format(options['net']))))
            print('step2, loading model')
            params_to_optimize = self._net.parameters()

        

        self._optimizer = torch.optim.SGD(params_to_optimize, lr=self._options['base_lr'], momentum=0.9,
                                              weight_decay=self._options['weight_decay'])
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=2, gamma=0.9)
        # Learning rate warm-up
        if self._warmup > 0:
            warmup = lambda epoch: epoch / 5
            self._warmupscheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda = warmup)
        else:
            print('no warmup')

        if self._options['cos'] == False:
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode='max', factor=0.1, patience=3, verbose=True, threshold=1e-4)
            print('lr_scheduler: ReduceLROnPlateau')
        else:
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=self._options['epochs'])
            print('lr_scheduler: CosineAnnealingLR')

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

        # Load data
        data_dir = self._data_base
        train_data = Imagefolder_modified(os.path.join(data_dir, 'train.txt'), transform=train_transform)
        self._train_num = len(train_data)
        # If you want to load images into RAM once, set 'cached=True'
        test_data = Imagefolder_modified(os.path.join(data_dir, 'val.txt'), transform=test_transform, cached=False)

        self._train_loader = DataLoader(train_data, batch_size=self._options['batch_size'],
                                        shuffle=True, num_workers=4, pin_memory=True)
        self._test_loader = DataLoader(test_data, batch_size=512,
                                       shuffle=False, num_workers=4, pin_memory=True)
        self._Cross_entropy_T = []
        self._logits_softmax_T_1 = [torch.ones(options['n_classes']).cpu() for j in range(len(train_data))]
        self._false_id = []

    # Compute probability cross-entropy
    def cross_entropy(self,P1, P2):
        P1 = P1.float()
        P2 = P2.float()
        return -torch.sum(P2 * torch.log(P1))

    # Global selection based on probability cross-entropy
    def selection_loss(self, logits_now, labels, ids):
        loss_update = [i for i in range(len(ids))]

        noise = list(set(self._false_id).intersection(set(ids.numpy())))
        for i in range(len(ids)):
            if ids[i] in noise:
                loss_update.remove(i)

        logits_final = logits_now[loss_update]
        labels_final = labels[loss_update]

        if self._smooth == True:
            loss = self._smooth_label_loss(logits_final, labels_final)
        else:
            loss = self._criterion(logits_final, labels_final)
        return loss

    # Loss function with label smoothing
    def _smooth_label_loss(self,logits,labels):
        N = labels.size(0)
        smoothed_labels = torch.full(size=(N, self._class),
                                     fill_value=(1 - self._label_weight) / (self._class - 1)).cuda()
        smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=self._label_weight)
        log_prob = torch.nn.functional.log_softmax(logits, dim=1)
        loss = -torch.sum(log_prob * smoothed_labels) / N
        return loss

    def train(self):
        """
        Train the network
        """
        print('Training ... ')
        logfile = 'train_log.txt'
        best_accuracy = 0.0
        best_epoch = 0
        print('Epoch\tTrain Loss\tTrain Accuracy\tTest Accuracy\tEpoch Runtime')
        _logits_softmax_T_1 = [torch.ones(options['n_classes']).cpu() for j in range(self._train_num)]
        for t in range(self._options['epochs']):
            if self._warmup >t:
                self._warmupscheduler.step()
                print('warmup learning rate',self._optimizer.state_dict()['param_groups'][0]['lr'])
            epoch_start = time.time()
            epoch_loss = []
            num_correct = 0
            num_total = 0
            num_remember = 0
            Cross_entropy_now = []
            steps = 0
            for X, y, id, path in self._train_loader:
                # Enable training mode
                self._net.train(True)
                # Data
                X = X.cuda()
                y = y.cuda()
                # Forward pass
                score = self._net(X)  # score is in shape (N, 200)
                if self._denoise and t > 1:
                    # Global Selection
                    loss = self.selection_loss(score, y,id)
                else:
                    if self._smooth == False:
                        loss = self._criterion(score, y)
                    else:
                        # Smooth label loss
                        loss = self._smooth_label_loss(score, y)

                epoch_loss.append(loss.item())
                # Prediction
                closest_dis, prediction = torch.max(score.data, 1)

                logits_softmax = F.softmax(score, dim=1).clone().detach()
                # logits_softmax = F.softmax(score, dim=1)
                for i in range(id.shape[0]):
                    # Compute probability cross-entropy
                    ce = self.cross_entropy(_logits_softmax_T_1[id[i]], logits_softmax[i].detach().cpu())
                    # ce = torch.rand(1)
                    # Record softmax probability
                    _logits_softmax_T_1[id[i]] = logits_softmax[i].cpu()
                    tmp = []
                    tmp.append(id[i].clone().cpu())
                    tmp.append(ce)
                    Cross_entropy_now.append(tmp)

                # prediction is the index location of the maximum value found,
                num_total += y.size(0)  # y.size(0) is the batch size
                num_correct += torch.sum(prediction == y.data).item()

                # Clear the existing gradients
                self._optimizer.zero_grad()
                # Backward
                loss.backward()
                self._optimizer.step()
                steps += 1
                if(steps % 370 == 0):
                    print('train : %d\t%4.2f%%\t\t%4.3f\t\t%4.2f%%\t\t%s' % (t + 1,steps * 100 * 128 / 473587,  loss.item(),torch.sum(prediction == y.data).item() / y.size(0),time.asctime(time.localtime())))
            # Record the train accuracy of each epoch
            train_accuracy = 100 * num_correct / num_total
            test_accuracy = self.test(self._test_loader)
            if self._warmup <= t:
                if self._options['cos'] == False:
                    self._scheduler.step(test_accuracy)  # the scheduler adjust lr based on test_accuracy
                else:
                    self._scheduler.step()

            Cross_entropy_now.sort(key=lambda x: x[1])  # 升序
            self._Cross_entropy_T = Cross_entropy_now.copy()

            if self._denoise == True and len(self._Cross_entropy_T) > 0:
                all_id = [int(x[0]) for x in self._Cross_entropy_T]
                num_remember = int((1 - min(t / self._tk, 1) * self._drop_rate) * len(all_id))
                self._false_id = all_id[num_remember:]

            epoch_end = time.time()
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = t +1
                print('*', end='')
                # Save mode
                # if options['net'] == 'bcnn':
                    # Save mode for each step
                torch.save(self._net.state_dict(), os.path.join(self._path, options['net'] + '_step{}.pth'.format(self._step)))
                # else:
                    # torch.save(self._net.state_dict(), os.path.join(self._path, options['net'] + '.pth'))
            # self.scheduler.step()
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f\t\t%4.2f\t\t%5f' % (t + 1, sum(epoch_loss) / len(epoch_loss),
                                                            train_accuracy, test_accuracy,
                                                            epoch_end - epoch_start, num_remember, self._optimizer.param_groups[0]["lr"] ))
            
                                                          
            with open(logfile, "a") as f:
                output = '%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f\t\t%4.2f\t\t%5f' % (t + 1, sum(epoch_loss) / len(epoch_loss),
                                                            train_accuracy, test_accuracy,
                                                            epoch_end - epoch_start, num_remember, self._optimizer.param_groups[0]["lr"])
                f.write(output + "\n")


        print('******\n'
        'Best Accuracy 1: [{0:6.2f}%], at Epoch [{1:03d}] '
        '\n******'.format(best_accuracy, best_epoch))
        with open(logfile, "a") as f:
            output = '******\n' \
                    'Best Accuracy 1: [{0:6.2f}], at Epoch [{1:03d}]; ' \
                    '\n******'.format(best_accuracy, best_epoch)
            f.write(output + "\n")
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

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--net', dest='net', type=str, default='resnet50',
                        help='supported options: resnet152, resnet50')
    parser.add_argument('--n_classes', dest='n_classes', type=int, default=100,
                        help='number of classes')
    # Path to save model
    parser.add_argument('--path', dest='path', type=str, default='model')
    # Pretrained model path
    parser.add_argument('--prt_model', dest='prt_model', type=str)
    # Training and test data path
    parser.add_argument('--data_base', dest='data_base', type=str)
    # Learning rate, weight decay, epochs and batch size
    parser.add_argument('--lr', dest='lr', type=float, default=1e-2)
    parser.add_argument('--w_decay', dest='w_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', dest='epochs', type=int, default=80)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    # Turn on denoising, label smoothing
    parser.add_argument('--denoise', action='store_true', help='Turns on denoising', default=False)
    parser.add_argument('--drop_rate', type=float, default=0.25)
    parser.add_argument('--smooth', action='store_true', help='Turns on smooth label', default=False)
    parser.add_argument('--label_weight', dest='label_weight', type=float, default=0.5)
    # Other settings
    parser.add_argument('--cos', action='store_true', help='Turns on cos learning rate', default=False)
    parser.add_argument('--tk', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--step', dest='step', type=int, default=1,
                        help='Step 1 is training fc only; step 2 is training the entire network')

    args = parser.parse_args()

    model = args.path

    print(os.path.join(os.popen('pwd').read().strip(), model))

    if not os.path.isdir(os.path.join(os.popen('pwd').read().strip(), model)):
        print('>>>>>> Creating directory .\'model\' ... ')
        os.mkdir(os.path.join(os.popen('pwd').read().strip(), model))

    path = os.path.join(os.popen('pwd').read().strip(), model)

    options = {
            'base_lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'weight_decay': args.w_decay,
            'path': path,
            'data_base': args.data_base,
            'net': args.net,
            'n_classes': args.n_classes,
            'denoise': args.denoise,
            'drop_rate': args.drop_rate,
            'smooth': args.smooth,
            'label_weight': args.label_weight,
            'cos':args.cos,
            'tk': args.tk,
            'warmup': args.warmup,
            'step': args.step,
            'prt_model':args.prt_model
        }
    manager = Manager(options)
    manager.train()