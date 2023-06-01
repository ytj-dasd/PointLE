import torch
import numpy as np
import os
from torch import optim,nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class ObjectDataset(Dataset):
    def __init__(self, path, delimiter = ' '):
        self.__data = np.genfromtxt(path, delimiter = delimiter).astype(np.float32)

    def __getitem__(self, index):
        instance = self.__data[index,:]
        # data = torch.zeros(15,769).float()
        # single_data = np.zeros(769)
        # for idx in range(15):
        #     single_data[0:768] = instance[idx*768:(idx+1)*768]
        #     single_data[768] = instance[11520+idx]
        data = torch.zeros(15,769).float()
        single_data = np.zeros(769)
        for idx in range(15):
            single_data[0:768] = instance[idx*768:idx*768+768]
            single_data[768] = instance[11520+idx]
            #print(single_data)
            data[idx,:] = torch.FloatTensor(single_data)
        label = torch.FloatTensor(instance[11535:11538])
        return data,label

    def __len__(self):
        return self.__data.shape[0]


