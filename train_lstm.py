import os
import sys
import argparse
import time
import random
import pdb

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_load import ObjectDataset
from model import LSTMClassifier

min_score = 0.7

class BCEFocalLoss(nn.Module):
  """
  二分类的Focalloss alpha 固定
  """
  def __init__(self, gamma=0.6, alpha=0.7, reduction='mean'):
    super(BCEFocalLoss,self).__init__()
    self.gamma = gamma
    self.alpha = alpha
    self.reduction = reduction
  
  def forward(self, _input, target):
    alpha = self.alpha
    _input = _input.clamp(min = 0.0001 , max = 0.9999)
    loss = - alpha * (1 - _input) ** self.gamma * target * torch.log(_input) -(1 - alpha) * _input ** self.gamma * (1 - target) * torch.log(1 - _input)
    if self.reduction == 'mean':
      loss = torch.mean(loss)
    elif self.reduction == 'sum':
      loss = torch.sum(loss)
    return loss

def test(model,filename):
    test_dataloader = DataLoader(dataset=ObjectDataset(filename),
                            batch_size=8,
                            shuffle=False)
    total = 0
    good = 0
    for instance, labels in test_dataloader: 
        instance, labels = instance.to(device), labels.to(device)
        output, total_output = model(instance)   
        good = good + sum(row.all().int().item() for row in (output.ge(min_score) == labels))
        total = total + len(instance)
    return float(good),float(total)

def visual_ouput(model):
    test_dataloader = DataLoader(dataset=ObjectDataset('pre_process/new_dataset/val.txt'),
                            batch_size=8,
                            shuffle=False)
    f = open(os.path.join('output.txt'), 'w')
    for instance, labels  in test_dataloader: 
        instance, labels = instance.to(device), labels.to(device)
        output, total_output = model(instance)   
        for j in range(len(instance)):
            f.write('label: ' + str(labels[j]) +' output: ' + str(output[j]) + '\n')
            for k in range(total_output.size(0)):
                f.write(str(k) + ' output: ' + str(total_output[k][j]) + '\n')
        f.write('\n')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=512,
										    help='LSTM hidden dimensions')
    parser.add_argument('--batch_size', type=int, default=8,
											help='size for each minibatch')
    parser.add_argument('--num_epochs', type=int, default=200,
											help='maximum number of epochs')
    parser.add_argument('--char_dim', type=int, default=128,
											help='character embedding dimensions: 128/256/512')
    parser.add_argument('--learning_rate', type=float, default=0.004,
											help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
											help='weight_decay rate')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #fout = open(os.path.join('loss_accuracy.txt'),'w')
    model = LSTMClassifier(args.char_dim, args.hidden_dim, 3)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = BCEFocalLoss()
    #criterion = nn.BCELoss()
    dataloader = DataLoader(dataset = ObjectDataset('dataset/train.txt'),
                        batch_size = args.batch_size,
                        shuffle = True)

    for epoch in range(args.num_epochs):
        total_data = 0
        good_pred = 0
        running_loss = 0
        for instances, labels  in dataloader:
            instances, labels = instances.to(device), labels.to(device)
            #print(instances)
            optimizer.zero_grad()
            
            #print(instances.size())
            output, total_output = model(instances)
            #print(instances.shape,output.shape,labels.shape)
            loss = criterion(output, labels)
            
            #result = torch.argmax(output,dim=1)
            good_pred = good_pred + sum(row.all().int().item() for row in (output.ge(min_score) == labels))
            total_data = total_data + len(instances)    
            running_loss += loss.item()
            #total_num = sum(p.numel() for p in model.parameters())
            #print('num: ',total_num)

            loss.backward()
            optimizer.step()
        #if epoch==150:
                #visual_ouput(model)
        #test_acc = test(model,'pre_process/ml_test.txt')
        good_car,total_car = test(model,'dataset/test_car.txt') 
        good_pedestrain,total_pedestrain = test(model,'dataset/test_pedestrain.txt') 
        good_bic,total_bic = test(model,'dataset/test_bicycle.txt')
        car_acc = good_car / total_car
        pedestrain_acc = good_pedestrain / total_pedestrain
        biy_acc = good_bic / total_bic
        test_acc = (good_bic+good_car+good_pedestrain)/(total_bic+total_car+total_pedestrain)

        # val_good_car,val_total_car = test(model,'pre_process/new_dataset/val_car.txt') 
        # val_good_pedestrain,val_total_pedestrain = test(model,'pre_process/new_dataset/val_pedestrain.txt') 
        # val_good_bic,val_total_bic = test(model,'pre_process/new_dataset/val_bicycle.txt')
        # val_car_acc = val_good_car / val_total_car
        # val_pedestrain_acc = val_good_pedestrain / val_total_pedestrain
        # val_biy_acc = val_good_bic / val_total_bic
        # val_acc = (val_good_bic+val_good_car+val_good_pedestrain)/(val_total_bic+val_total_car+val_total_pedestrain)

        print('epoch: ',epoch, ' loss: ',running_loss / len(dataloader),' train_acc: ', float(good_pred / total_data) , ' test_acc: ', test_acc, ' car_acc: ', car_acc, ' pedestrain_acc: ', pedestrain_acc , ' biy_acc: ', biy_acc)
        #print(total_data,total_car,total_pedestrain,total_bic)
        #print(' val_acc: ', val_acc, ' val_car_acc: ', val_car_acc, ' val_pedestrain_acc: ', val_pedestrain_acc , ' val_biy_acc: ', val_biy_acc)
        print(' ')
        #fout.write(str(epoch) + ' ' + str(running_loss / len(dataloader)) + ' ' + str(float(good_pred / total_data)) + ' ' +  str(test_acc) + '\n')
    #torch.save(model,'/home/ytj/文档/mlp-pytorch/model/model.pth')