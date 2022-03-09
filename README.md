# DeblurGAN-C
This repository contains the code used in [DeblurGAN-C: image restoration using GAN and a correntropy based loss function in degraded visual environments](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11395/1139507/DeblurGAN-C--image-restoration-using-GAN-and-a-correntropy/10.1117/12.2560792.short?SSO=1)

-- Download HR Image Dataset


-- Generate training dataset


-- Train network

python scripts_process/train.py --input_folder images/Input --n_images_train 50 --n_images_valid 10 --batch_size 5 --epoch_num 10 --critic_updates 5 --loss_type 5 --loss_weight 100

-- Run validation

python scripts_process/valid.py --input_dir=images/Input_valid --target_dir=images/Target_valid --weight_path=weights/generator.h5 --output_dir Enhanced_Valid

