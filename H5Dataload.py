import os
import torch
import numpy as np  
import h5py
from torch.utils.data import Dataset 


type = torch.float32

def ReadTrainingData(datapath):
    fileRelist = []
    fileImlist = []
    datalist = os.listdir(datapath)
    for n in range (len(datalist)):
        fileRe = os.path.join(datapath,"skin_fringe_%d.h5"%(n+1))
        fileRelist.append(fileRe)
    return fileRelist


class ReadTrainingDataFromH5(Dataset):
    def __init__(self, datapath, transform):
        self.datadir = datapath
        self.datalistRe = ReadTrainingData(self.datadir)
        self.transform = transform
        
    def __getitem__(self, index):
        datanameRe = self.datalistRe[index]
        f = h5py.File(datanameRe, 'r+')
        fringedataset = f['fringe']
        fringe = np.zeros([384,256] ) 

        for index in range(len(fringedataset)):
            fringe[:,index] = fringedataset[index]
        fringe_crop = fringe[:,:]
        if self.transform is not None:
           fringeRe_tensor = self.transform(fringe_crop)
        return fringeRe_tensor

    def __len__(self):
        return len(self.datalistRe)
    
def ReadValidationData(datapath):
    fileRelist = []
    fileImlist = []
    datalist = os.listdir(datapath)
    for n in range (len(datalist)):
        fileRe = os.path.join(datapath,"skin_fringe_%d.h5"%(n+17512)) 
        # If testing, the index should be changed according to the index of testing data
        fileRelist.append(fileRe)

    return fileRelist


class ReadValidationDataFromH5(Dataset):
    def __init__(self, datapath, transform):
        self.datadir = datapath
        self.datalistRe = ReadValidationData(self.datadir)
        self.transform = transform
        
    def __getitem__(self, index):
        datanameRe = self.datalistRe[index]
        f = h5py.File(datanameRe, 'r+')
        fringedataset = f['fringe']
        fringe = np.zeros([384,256]) 

        for index in range(len(fringedataset)):
            fringe[:,index] = fringedataset[index]
        fringe_crop = fringe
        if self.transform is not None:
           fringeRe_tensor = self.transform(fringe_crop)
        return fringeRe_tensor

    def __len__(self):
        return len(self.datalistRe)
    