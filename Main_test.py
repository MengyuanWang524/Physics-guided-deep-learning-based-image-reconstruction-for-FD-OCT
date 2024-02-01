import os
import numpy as np 
from numpy import fft, cos, inf, save, savetxt, zeros, array, exp, conj, nan, isnan, pi, sin 
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import  transforms
from torch_poly_lr_decay import PolynomialLRDecay
from H5Dataload import ReadTrainingDataFromH5,ReadValidationDataFromH5
from unet import UNet


dtype = torch.float32

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## Read test data

data_path_test  ='Test'

transform_totensor = transforms.Compose([transforms.ToTensor(),transforms.ConvertImageDtype(dtype)])


M = 384

# Reconstruction grid
T = M      

lambda0 =1310e-9

FWHM_lambda = 50e-9

lambda_st = 1260e-9

lambda_end = 1360e-9


Dlambda = lambda_end - lambda_st

k0 = 2 * np.pi / lambda0

k_st = 2 * np.pi / lambda_end

k_end = 2 * np.pi / lambda_st

dk = (k_end - k_st) / M

dz_fft =  np.pi / (dk * M) 


dz = dz_fft

ref = 0


k = np.linspace(k_st, k_st + (M - 1) * dk, M )

gridRec = np.linspace(0, (T - 1) * dz, T ) #- (T / 2 ) * dz


X, Y = np.meshgrid(gridRec, k)

matFourRec = exp(- 2 * 1j * X * Y)


matFourRec_tensorRe = torch.tensor(matFourRec.real, dtype=dtype).to(device)


model = UNet().to(device)

## Load the model ##    

model.load_state_dict(torch.load("skin_Unet.pth",map_location=device))

loss_func = nn.MSELoss().to(device)

OCT_dataset_test= ReadValidationDataFromH5(datapath = data_path_test, transform=transform_totensor)

Valid_loader = DataLoader(OCT_dataset_test, batch_size = 1, shuffle= True)


##############
#     Test   #
##############

model.eval()
step = 0

for f1 in Valid_loader:

    input = f1.to(device) 

    t1 = time.time()
    pred_obj = model(input)
    t2 = time.time()
    
    pred_f1 =  torch.matmul(matFourRec_tensorRe, pred_obj)


    fringe_est = pred_f1.squeeze().cpu().detach().numpy()
    np.savetxt(fname = 'fringe_test_unet_%d.csv'
                    %( step), X = fringe_est, delimiter=',')
    
    image_est= pred_obj.squeeze().cpu().detach().numpy()
    np.savetxt(fname = 'Image_test_unet_%d.csv'
                %(step ), X = image_est, delimiter=',')
    
    runtime = t2-t1
    print("Time for each image:%f" %runtime)
    print("MSE loss:%f"%loss_func(pred_f1,input))
    print(step)