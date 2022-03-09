# DeblurGAN-C
This repository contains the code used in [DeblurGAN-C: image restoration using GAN and a correntropy based loss function in degraded visual environments](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11395/1139507/DeblurGAN-C--image-restoration-using-GAN-and-a-correntropy/10.1117/12.2560792.short?SSO=1)

**Download HR Image Dataset**

Download the High resolution DIV2K and/or Flickr2K dataset set from https://drive.google.com/drive/folders/1B-uaxvV9qeuQ-t7MFiN1oEdA6dKnj2vW 

Add training images to images/Train_HR

Add validation images to images/Valid_HR

**Generate Training Dataset**

Run the MATLAB file: Matlab_scripts/Gen_rand_blur_noise.m

Parameters for generating the degraded images can be found on lines 4-36.
Images are saved to images folder.

**Train Network**

python python_scripts/train.py --input_folder images/Input --n_images_train 50 --n_images_valid 10 --batch_size 5 --epoch_num 10 --critic_updates 5 --loss_type 5 --loss_weight 100

**Run Validation**

python python_scripts/valid.py --input_dir=images/Input_valid --target_dir=images/Target_valid --weight_path=weights/generator.h5 --output_dir Enhanced_Valid

