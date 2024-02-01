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

##############
#   Train   #
##############

dtype = torch.float32

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## Read training and validation data from the files
data_path_train ='/mnt/data/mengyuan/Train_clean_new'

data_path_valid = '/mnt/data/mengyuan/Valid_clean_new'


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


def TotalVariationLoss(inputImg):
    # inputImg = 10* torch.log10(abs(inputImg))
    h_r = inputImg.size(dim = -1)
    w_r = inputImg.size(dim = -2)
    tv_h = torch.pow( inputImg[:,:, 1:,:] - inputImg[:,:,:-1,:], 2 ).sum()
    tv_w = torch.pow( inputImg[:,:, :,1:] - inputImg[:,:,:,:-1], 2 ).sum()
    return (tv_h+tv_w)/(h_r*w_r)

def NPCC_Loss( T , Y ):
    # Calculate NPCC.
         
    m,n,R,N = Y.shape

    c = R*N

    T = T.reshape(m,n,c)

    Y = Y.reshape(m,n,c)
    
                       
    T0 = T - torch.mean(T)
    Y0 = Y - torch.mean(Y)

    T0_norm = torch.sqrt(torch.sum(T0 **2))

    Y0_norm = torch.sqrt(torch.sum(Y0 **2))    

    npcc = - ( torch.sum(T0*Y0) )/ ( T0_norm*Y0_norm )

    return npcc


## Hyper parameters ##
batch_size = 8
epoch = 1000
learning_rate = 1e-3
lamdaTV = 0.01


def train_model( model, loss_func, lamdaTV, optimizer, num_epochs, device,
                dataload_train, dataload_valid, matFourRe):
    
    train_losses = zeros([num_epochs])
    valid_losses = zeros([num_epochs])
    MSElossRe = zeros([num_epochs])
    NPCCloss = zeros([num_epochs])
    TVloss = zeros([num_epochs])

    scheduler = optim.lr_scheduler.StepLR(optimizer, 200, 0.5)
    scheduler = PolynomialLRDecay(optimizer, max_decay_steps=num_epochs-1, end_learning_rate= 1e-4, power=1.0)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_datasize = len(dataload_train.dataset)
        valid_datasize = len(dataload_valid.dataset)
        Train_loss = 0
        Valid_loss = 0
        MSE_loss = 0
        NPCC_loss = 0
        TV_loss = 0
        step = 0
        step2 = 0
        t1 = time.time()

        model.train()
        for x in dataload_train:
            step += 1

            fringeRe = x.to(device)
            
            fringe_measurement = fringeRe

            optimizer.zero_grad()

            estimation_obj= model(fringe_measurement)

            fringeRe_estimation = torch.matmul(matFourRe, estimation_obj)

            fringe_loss = loss_func(fringeRe, fringeRe_estimation)



            im_loss_tv = lamdaTV* TotalVariationLoss(estimation_obj)

            loss = fringe_loss + im_loss_tv 
    
            loss.backward()


            optimizer.step()
            
            
            Train_loss += loss.item()

            MSE_loss += fringe_loss.item()
            

            TV_loss += im_loss_tv.item()

            print("%d/%d, Real_loss:%0.8f, T V_loss:%0.8f" 
                  % (step, (train_datasize - 1) // dataload_train.batch_size + 1, 
                     fringe_loss.item(),  im_loss_tv.item()))

        t2 = time.time()
        runtime = t2-t1
        print("Time for each epoch:%f" %runtime)

        print("epoch %d MSE ReLoss:%0.8f TV Loss:%0.8f" 
              % (epoch, MSE_loss/step,  TV_loss/step))
        
        train_losses[epoch] = Train_loss /step
        MSElossRe[epoch] = MSE_loss /step
        NPCCloss[epoch] = NPCC_loss/step 
        TVloss[epoch] = TV_loss /step
        
        scheduler.step()
        

        for x1 in dataload_valid:
            step2 += 1

            inp1= x1.to(device)   

            input = inp1

            pred_obj = model(input)

            pred_f1 =  torch.matmul(matFourRe, pred_obj)

            fringeRe_loss2 = loss_func(pred_f1, inp1)

            fringe_loss2 = fringeRe_loss2      

            im_loss2_tv  = lamdaTV* TotalVariationLoss(pred_obj)

            validloss =  fringe_loss2 + im_loss2_tv            

            Valid_loss += validloss.item()

            print("%d/%d,valid_loss:%0.8f" 
                  % (step2, (valid_datasize - 1) // dataload_valid.batch_size + 1, validloss.item()))
    
        print("epoch %d Valid Loss:%0.8f" % (epoch, Valid_loss/step2))

        valid_losses[epoch] = Valid_loss / step2

        ### Save model####           
    torch.save(model.state_dict(), 'skin_epoch%d_%f_dz%f_%d_lamda_%d_lamdaTV_%d.pth' 
         %(epoch,learning_rate,dz, num_epochs, lamdaTV))   
        
    np.savetxt(fname = 'train_skin_%f_dz%f_%d_lamda_%d_lamdaTV_%d.csv'
        % (learning_rate,dz,num_epochs,lamdaTV), X = train_losses, delimiter=',')
    
    np.savetxt(fname = 'mseRe_skin_%f_dz%f_%d_lamda_%d_lamdaTV_%d.csv'
        % (learning_rate,dz,num_epochs,lamdaTV), X = MSElossRe, delimiter=',')
    
    np.savetxt(fname = 'npcc_skin_%f_dz%f_%d_lamda_%d_lamdaTV_%d.csv'
                % (learning_rate,dz,num_epochs,lamdaTV), X = NPCCloss, delimiter=',')
        
    np.savetxt(fname = 'tv_skin_%f_dz%f_%d_lamda_%d_lamdaTV_%d.csv'
        % (learning_rate,dz,num_epochs,lamdaTV), X = TVloss, delimiter=',')
        
    np.savetxt(fname = 'valid_skin_%f_dz%f_%d_lamda_%d_lamdaTV_%d.csv'
        % (learning_rate,dz,num_epochs,lamdaTV), X = valid_losses, delimiter=',') 

    return model


####################
# Model training  #
####################

model = UNet().to(device)

## Load the model ##    

# model.load_state_dict(torch.load("skin_Unet.pth",map_location=device))

loss_func = nn.MSELoss().to(device)

optimizer = optim.Adam(model.parameters(),lr = learning_rate) 

OCT_dataset_train = ReadTrainingDataFromH5(datapath = data_path_train, transform=transform_totensor)

Train_loader = DataLoader(OCT_dataset_train, batch_size = batch_size, shuffle= True)

OCT_dataset_validation = ReadValidationDataFromH5(datapath = data_path_valid, transform=transform_totensor)

Valid_loader = DataLoader(OCT_dataset_validation, batch_size = batch_size, shuffle= True)


train_model(model, loss_func = loss_func, lamdaTV = lamdaTV, optimizer=optimizer,num_epochs = epoch, device=device,
             dataload_train = Train_loader,dataload_valid= Valid_loader, 
             matFourRe = matFourRec_tensorRe)

