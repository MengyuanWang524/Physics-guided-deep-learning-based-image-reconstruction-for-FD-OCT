# Physics-guided-deep-learning-based-image-reconstruction-for-FD-OCT
This is the code for the paper: Physics-guided deep learning-based image reconstruction for Fourier-domain optical coherence tomography
The code is based on Python 3.7

1. Training
To train a network, run:
Main_train.py

nohup python train_OCT_skin_Unet_384.py >result/result.log 2>&1 &

