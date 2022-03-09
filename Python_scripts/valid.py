import numpy as np
from PIL import Image
from os import listdir, makedirs
from os.path import exists, join, basename, isfile
import click
from SSIM_PIL import compare_ssim

from model import *
from utils import *
import time

def test(input_dir,target_dir,weight_path,output_dir):
    g = generator_model()
    g.load_weights(weight_path)
    name=os.path.splitext(basename(weight_path))[0]
    
    if not exists(output_dir):
        makedirs(output_dir)
    
    input_folder=join(output_dir,'input')
    target_folder=join(output_dir,'target')
    #if not exists(input_folder):
        #makedirs(input_folder)
    #if not exists(target_folder):
        #makedirs(target_folder)
    
    all_input_paths, all_target_paths = list_image_files(input_dir), list_image_files(target_dir)
    
    PT=[]
    PSNR=[]
    SSIMS=[]
    for input_path, target_path in zip(all_input_paths, all_target_paths):
        if isfile(input_path):
            start= time.time()
            
            input_img, target_img = load_image(input_path,target_path)
            
            
            x_test = np.array([preprocess_image(input_img)])
                
            generated = g.predict(x=x_test)

            generated_image=(deprocess_image(generated))
            generated_image=np.squeeze(generated_image,0)
            
            end = time.time()
            
            PT.append(end-start)
            #print('Process Time = {} seconds'.format(end-start))
            
            #MSE=np.mean((generated_image-np.array(target_img))**2)
            #PSNR.append(10*np.log10((255**2)/MSE))
            #print('PSNR of Generated Image is {}'.format(10*np.log10((255**2)/MSE)))
            
            generated_image = Image.fromarray(generated_image.astype(np.uint8))
            
            #ssim=compare_ssim(target_img,generated_image)
            #SSIMS.append(ssim)
            #print('SSIM of Generated Image is {}'.format(ssim))
            
            #input_img.save(join(input_folder,'input_'+basename(input_path))) 
            #target_img.save(join(target_folder,'target_'+basename(input_path))) 
            generated_image.save(join(output_dir,'generated_'+name+'_'+basename(input_path)))
            print('Output Saved as ' + output_dir + '/generated_'+name+'_'+basename(input_path))
            
            
    #PT_avg=np.mean(PT)
    #print('Average Process Time = {} seconds'.format(PT_avg))
    #PSNR_avg=np.mean(PSNR)
    #PSNR_min=np.min(PSNR)
    #PSNR_max=np.max(PSNR)
    #print('Average PSNR of Generated Images is {}'.format(PSNR_avg))
    #print('Minimum PSNR of Generated Images is {}'.format(PSNR_min))
    #print('Maximum PSNR of Generated Images is {}'.format(PSNR_max))
    #SSIMS_avg=np.mean(SSIMS)
    #SSIMS_min=np.min(SSIMS)
    #SSIMS_max=np.max(SSIMS)
    #print('Average SSIM of Generated Images is {}'.format(SSIMS_avg))
    #print('Minimum SSIM of Generated Images is {}'.format(SSIMS_min))
    #print('Maximum SSIM of Generated Images is {}'.format(SSIMS_max))
        
@click.command()
@click.option('--input_dir', help='Input Image to deblur')
@click.option('--target_dir', help='Reference Target Image')
@click.option('--weight_path', help='Weights to apply')
@click.option('--output_dir', help='Output File Directory')

def test_command(input_dir,target_dir,weight_path,output_dir):
    return test(input_dir,target_dir,weight_path,output_dir)


if __name__ == "__main__":
    test_command()

