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
    if not exists(output_dir):
        makedirs(output_dir)
    
    PT=[];
    PSNR=[];
    PSNR_patches=[];
    SSIMS=[];
    for f in listdir(input_dir):
        if isfile(join(input_dir,f)):
            input_path=join(input_dir,f)
            target_path=join(target_dir,f)
            
            start1= time.time()
            
            input_patches, dimW, dimH = load_test_patches_2(input_path)
            
            generated_images = []
            
            for patch in zip(input_patches):
                x_test = np.array(preprocess_image(patch))
                
                generated = g.predict(x=x_test)

                generated_images.extend(deprocess_image(generated))
                
            generated_image=reconstruct_image_2(generated_images,dimW,dimH)                
                        
            end1 = time.time()
            PT.append(end1-start1)
            #print('Process_Time = {} seconds'.format(end1-start1))
            
            #target_patches, dimW, dimH = load_test_patches_2(target_path)
            
            input_img=reconstruct_image_2(input_patches,dimW,dimH)
            target_img=reconstruct_image_2(target_patches,dimW,dimH)
            
            #MSE_patches=np.mean((np.array(generated_images)-np.array(target_patches))**2)
            #PSNR_patches.append(10*np.log10((255**2)/MSE_patches))
            #print('PSNR of Generated Patches is {}'.format(10*np.log10((255**2)/MSE_patches)))
            
            #MSE=np.mean((generated_image-target_img)**2)
            #PSNR.append(10*np.log10((255**2)/MSE))
            #print('PSNR of Generated Image is {}'.format(10*np.log10((255**2)/MSE)))
            
            #for i in range(np.shape(target_patches)[0]):
            #    tar=Image.fromarray(target_patches[i].astype(np.uint8))
            #    gen=Image.fromarray(generated_images[i].astype(np.uint8))
            #    SSIMS.append(compare_ssim(tar,gen,GPU=False))     
                
            
            #input_image = Image.fromarray(input_img.astype(np.uint8))
            #target_image = Image.fromarray(target_img.astype(np.uint8))
            #generated_image = Image.fromarray(generated_image.astype(np.uint8))
            
            #ssim=compare_ssim(target_image,generated_image,GPU=False)
            #SSIMS.append(ssim)
            #print('SSIM of Generated Image is {}'.format(ssim))
            
            
            #input_image.save(join(output_dir,'input_'+basename(input_path))) 
            #target_image.save(join(output_dir,'target_'+basename(input_path))) 
            #generated_image.save(join(output_dir,'generated_'+basename(input_path)))  
            #print('Output Saved as ' + output_dir + '/generated_'+basename(input_path))
            
    #PT_avg=np.mean(PT)
    #print('Average Process Time = {} seconds'.format(PT_avg))

    #PSNR_patches_avg=np.mean(PSNR_patches)
    #PSNR_patches_min=np.min(PSNR_patches)
    #PSNR_patches_max=np.max(PSNR_patches)   
    #print('Average PSNR of Generated Patches is {}'.format(PSNR_patches_avg))
    #print('Minimum PSNR of Generated Patches is {}'.format(PSNR_patches_min))
    #print('Maximum PSNR of Generated Patches is {}'.format(PSNR_patches_max))
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

