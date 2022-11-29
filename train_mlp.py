import torch
import numpy as np
import os
import time
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

min_score = 0.7

class IrisDataset(Dataset):
    def __init__(self, path, delimiter=' '):
        self.__data = np.genfromtxt(path, delimiter=delimiter).astype(np.float32)

    def __getitem__(self, index):
        instance = self.__data[index,:]
        #data = torch.from_numpy(instance[:-3])
        #label = torch.from_numpy(instance[135:138])
        data = torch.FloatTensor(instance[:150])
        #distance = torch.FloatTensor(instance[135:150])
        label = torch.FloatTensor(instance[150:153])

        return data, label

    def __len__(self):
        return self.__data.shape[0]

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

#1.mlp层数和神经元个数设置
        self.fc1 = nn.Linear(150,200)
        self.fc2 = nn.Linear(200,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,3)
#2.sigmoid激活函数
    def forward(self, x):
        #x = torch.sigmoid(self.fc1(x)) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))


class BCEFocalLoss(nn.Module):
  """
  二分类的Focalloss alpha 固定
  """
  def __init__(self, gamma=1.0, alpha=0.25, reduction='mean'):
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
    test_dataloader = DataLoader(dataset=IrisDataset(filename),
                            batch_size=8,
                            shuffle=False)
    total = 0
    good = 0
    for instance, labels in test_dataloader: 
        instance, labels = instance.to(device), labels.to(device)
        output = model(instance)   
        good = good + sum(row.all().int().item() for row in (output.ge(min_score) == labels))
        total = total + len(instance)
    return float(good),float(total)

def visual_ouput(model):
    test_dataloader = DataLoader(dataset=IrisDataset('pre_process/test.txt'),
                            batch_size=8,
                            shuffle=False)
    f = open(os.path.join('visual_output.txt'), 'w')
    for instance, labels in test_dataloader: 
        instance, labels = instance.to(device), labels.to(device)
        pred_val = model(instance)   
        #result = F.softmax(pred_val,dim=1)
        #output = torch.argmax(result,dim=1)
        for j in range(len(instance)):
            f.write('label: ' + str(labels[j]) +' output: ' + str(pred_val[j]) + '\n')

if __name__ == '__main__':
    dataloader = DataLoader(dataset=IrisDataset('pre_process/train.txt'),
                            batch_size=8,
                            shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #fout = open(os.path.join('loss_accuracy.txt'), 'w')
    epochs = 150
    model = Classifier()
    model.to(device)
    #criterion = BCEFocalLoss()
    criterion = nn.BCELoss()
#3.Adams优化
    optimizer = optim.SGD(model.parameters(), lr=0.005) 
    
    t1 = time.time()
    for epoch in range(epochs):
        total_data = 0
        good_pred = 0
        running_loss = 0
        for instances, labels in dataloader:
            optimizer.zero_grad()
            instances, labels = instances.to(device), labels.to(device)
            #instances = instances.cuda()
            #labels = labels.cuda()
            output = model(instances)
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
        if epoch==149:
                visual_ouput(model)
        good_car,total_car = test(model,'pre_process/test_car.txt') 
        good_pedestrain,total_pedestrain = test(model,'pre_process/test_pedestrain.txt') 
        good_bic,total_bic = test(model,'pre_process/test_bicycle.txt')
        car_acc = good_car / total_car
        pedestrain_acc = good_pedestrain / total_pedestrain
        biy_acc = good_bic / total_bic
        test_acc = (good_bic+good_car+good_pedestrain)/(total_bic+total_car+total_pedestrain)
        print('epoch: ',epoch, ' loss: ',running_loss / len(dataloader),' train_acc: ', float(good_pred / total_data) , ' test_acc: ', test_acc, ' car_acc: ', car_acc, ' pedestrain_acc: ', pedestrain_acc , ' biy_acc: ', biy_acc)
        #fout.write(str(epoch) + ' ' + str(running_loss / len(dataloader)) + ' ' + str(float(good_pred / total_data)) + ' ' +  str(test_acc) + '\n')
    #torch.save(model,'/home/ytj/文档/mlp-pytorch/model/model.pth')
    #print('run time: ',time.time()-t1)
