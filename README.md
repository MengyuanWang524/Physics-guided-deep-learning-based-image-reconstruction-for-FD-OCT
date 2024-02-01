# Physics-guided-deep-learning-based-image-reconstruction-for-FD-OCT
This is the code for the paper: Physics-guided deep learning-based image reconstruction for Fourier-domain optical coherence tomography
The code is based on Python 3.7

Data preparation:
The Training and validation data should have been uploaded in two folders. The data was saved in '.h5' format
The code that loads the data from the .h5 file is H5Dataload.py 

1. Train
To train a network, run:
Main_train.py
Optional if a printed result is desired: nohup python train_OCT_skin_Unet_384.py >result/result.log 2>&1 &
The Training and validation data should have been uploaded in two files.
The model is from the paper：https://opg.optica.org/ol/fulltext.cfm?uri=ol-48-3-759&id=525607

2. Test
Run: Main_test.py
The test data is provided in '\Test'
Change the image index in the code 'H5Dataload.py ' according to the image index for testing 
