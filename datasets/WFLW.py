import os
import torch
import torch.nn
from .datasetTools import getRawAnnosList, getItem


class WFLW256(torch.utils.data.Dataset):
    
    def __init__(self, mode='train', augment=False):
        super(WFLW256, self).__init__()
        self.augment = augment
        print(os.getcwd())
        self.annosList = getRawAnnosList(os.getcwd()+'/datasets/annos/WFLW256_'+mode+'.txt')

    def __len__(self):
        return len(self.annosList)

    def __getitem__(self, idx):
        return getItem(self.augment, self.annosList[idx])

