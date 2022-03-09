import os
import datetime
import click
import numpy as np
import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from utils import *
from losses import *
from model import *
import scipy.io as sio
from keras.layers import Input
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from layer_utils import *
from functools import partial

N_epochs_save=50
GRADIENT_PENALTY_WEIGHT = 10 
Image_Size=128
Scale=1
image_shape = (Image_Size, Image_Size, 3)

BASE_DIR = 'weights/'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

fig = plt.figure(figsize=(10,7))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
Epoch_Num = []
Avg_Train_loss = []
Avg_Valid_loss = []

def save_train_losses(train_losses,folder_name,loss_T):
    name=folder_name.split('.')

    filename = name[0]+'_'+loss_T+'_Train_losses.mat'
    
    sio.savemat(filename,{'train_losses':train_losses})
    
    return
    
def save_valid_images(images,epoch_number, folder_name,loss_T):
    name=folder_name.split('.')
    
    save_dir = name[0]+'_'+loss_T+'_Validation_Results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i in range(int(len(images))):
        Full_image=Image.fromarray(images[i])
        Full_image.save(os.path.join(save_dir,'Epoch_{}_valid_image_{}.png'.format(epoch_number,i)))
    
    return

def plot_progress(Train_loss,Valid_loss,Epoch_Number,folder_name,loss_T):   
    name=folder_name.split('.')
    Epoch_Num.append(Epoch_Number)
    Avg_Train_loss.append(Train_loss)
    Avg_Valid_loss.append(Valid_loss)
    

    ax1.plot(Epoch_Num,Avg_Train_loss)
    ax1.set_xlabel('Epoch Number')
    ax1.set_title('Average Training Loss') 
    
    ax2.plot(Epoch_Num,Avg_Valid_loss)
    ax2.set_xlabel('Epoch')
    ax2.set_title('Validation PSNR')
    plt.draw()
    plt.pause(1)
    plt.show
    
    plt.savefig(name[0]+'_'+loss_T+'_Train_Valid_stats.png')
    
    return

def save_all_weights(d, g, epoch_number, folder, current_loss, loss_T):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR+'_'+folder, '{}_{}_{}'.format(now.month, now.day,loss_T))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}_{}.h5'.format(epoch_number, current_loss,loss_T)), True)
    #d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)


def train_multiple_outputs(input_folder,n_images_train,n_images_valid, batch_size, log_dir, epoch_num, critic_updates,loss_type,loss_weight):
    data = load_images('./images', os.path.basename(input_folder),n_images_train, n_images_valid)
    y_train, x_train = data['B'], data['A']
    y_valid, x_valid = data['D'], data['C']
    
    g = generator_model()
    d = discriminator_model()
    
    #Compile Full Discriminator Model
    outputs_real = Input(shape=image_shape)
    inputs_real = Input(shape=image_shape)
    outputs_fake = g(inputs_real)
    d_outputs_fake = d(outputs_fake)
    d_outputs_real = d(outputs_real)
    
    averaged_samples = RandomWeightedAverage()([outputs_real, outputs_fake])
    averaged_samples_out = d(averaged_samples)
    
    partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=averaged_samples, gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    partial_gp_loss.__name__ = 'gradient_penalty'
    
    full_d = Model(inputs=[outputs_real,inputs_real],outputs=[d_outputs_real, d_outputs_fake, averaged_samples_out])
    
    full_d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d.trainable = True
    g.trainable = False
    full_dloss=[wasserstein_loss,wasserstein_loss,partial_gp_loss]
            
    full_d.compile(optimizer=full_d_opt, loss=full_dloss)
    d.trainable = False
    g.trainable = True
    
    #Compile Discriminator on Generator
    d_on_g = generator_containing_discriminator_multiple_outputs(g, d)
    d_on_g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    if loss_type == 1:
        loss = [MSE_loss, wasserstein_loss]
        loss_T = 'MSE_loss'
    if loss_type == 2:
        loss = [l1_loss, wasserstein_loss]
        loss_T = 'l1_loss'
    if loss_type == 3:
        loss = [correntropy_05_loss, wasserstein_loss]
        loss_T = 'correntropy_05_loss'
    if loss_type == 4:
        loss = [VGG16_loss, wasserstein_loss]
        loss_T = 'VGG16_loss'
    if loss_type == 5:
        loss = [VGG16_correntropy_05_loss, wasserstein_loss]
        loss_T = 'VGG16_correntropy_05_loss'
    if loss_type == 6:
        loss = [VGG19_loss, wasserstein_loss]
        loss_T = 'VGG19_loss'
    loss_weights = [loss_weight, 1] #Loss Weights [Perceptual,Advs]
    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
    d.trainable = True
    g.trainable = False

    output_one_batch = np.ones((batch_size, 1))
    output_zero_batch = np.zeros((batch_size, 1))
    
    folder_name=os.path.basename(input_folder) 
    name=folder_name.split('.')
    log_path = log_dir+name[0]
    tensorboard_callback = TensorBoard(log_path)
    
    for epoch in tqdm.tqdm(range(epoch_num)):
        permutated_indexes = np.random.permutation(x_train.shape[0])

        d_losses = []
        d_on_g_losses = []
        for index in range(int(x_train.shape[0] / batch_size)):
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
            image_blur_batch = x_train[batch_indexes]
            image_full_batch = y_train[batch_indexes]

            generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)

            for _ in range(critic_updates):
                d_losses.append(full_d.train_on_batch([image_full_batch,image_blur_batch], [output_one_batch,output_zero_batch,output_zero_batch]))

            d.trainable = False
            g.trainable = True

            d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_one_batch])
            d_on_g_losses.append(d_on_g_loss)

            d.trainable = True
            g.trainable = False
            
        MSE=[]
        images_valid=[]
        for i in range(int(x_valid.shape[0])):
            img_in=np.array([x_valid[i]])
            img_tar=np.array([y_valid[i]])
            img_gen=g.predict(x=img_in, batch_size=1)
            
            
            img_tar=deprocess_image(np.squeeze(img_tar,axis=0))
            img_gen=deprocess_image(np.squeeze(img_gen,axis=0))
            
            if epoch%N_epochs_save == 0 and epoch!=0:
                img_in=deprocess_image(np.squeeze(img_in,axis=0))
                if Scale > 1:
                    before_pad=int(np.floor(((Image_Size*Scale)-Image_Size)/2))
                    after_pad=int(np.ceil(((Image_Size*Scale)-Image_Size)/2))
                    img_in=np.pad(img_in,((before_pad,after_pad),(before_pad,after_pad),(0,0)),'constant',constant_values=0)
                
                images_valid.append(np.concatenate((img_tar,img_in,img_gen),axis=1))
            MSE.append(np.mean((img_gen-img_tar)**2))
        
        Train_loss=np.mean(d_on_g_losses)
        Valid_MSE=np.mean(MSE)
        Valid_PSNR=10*np.log10((255**2)/Valid_MSE)
        
        
        plot_progress(Train_loss,Valid_PSNR,epoch,os.path.basename(input_folder),loss_T)
        
        save_train_losses(Avg_Train_loss,os.path.basename(input_folder),loss_T)
        
        # write_log(tensorboard_callback, ['g_loss', 'd_on_g_loss'], [np.mean(d_losses), np.mean(d_on_g_losses)], epoch_num)
        print('D_loss: {} G_loss: {} Valid_PSNR: {}dB'.format(np.mean(d_losses), np.mean(d_on_g_losses),Valid_PSNR))
        with open('log.txt', 'a+') as f:
            f.write('{} - {} - {}\n'.format(epoch, np.mean(d_losses), np.mean(d_on_g_losses)))
        if epoch%N_epochs_save == 0:
            save_valid_images(images_valid,epoch,os.path.basename(input_folder),loss_T)
            save_all_weights(d, g, epoch, os.path.basename(input_folder),int(np.mean(d_on_g_losses)),loss_T)

@click.command()
@click.option('--input_folder', help='Input image folder')
@click.option('--n_images_train', default=-1, help='Number of images to load for training')
@click.option('--n_images_valid', default=-1, help='Number of images to load for validation')
@click.option('--batch_size', default=10, help='Size of batch')
@click.option('--log_dir', default='./logs', help='Path to the log_dir for Tensorboard')
@click.option('--epoch_num', default=4, help='Number of epochs for training')
@click.option('--critic_updates', default=5, help='Number of discriminator training')
@click.option('--loss_type', default=2,help='Type of Loss')
@click.option('--loss_weight', default=100, help='Content Loss Weight')

def train_command(input_folder,n_images_train,n_images_valid, batch_size, log_dir, epoch_num, critic_updates,loss_type,loss_weight):
    return train_multiple_outputs(input_folder,n_images_train,n_images_valid, batch_size, log_dir, epoch_num, critic_updates,loss_type,loss_weight)


if __name__ == '__main__':
    train_command()
