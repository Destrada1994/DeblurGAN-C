import numpy as np
from PIL import Image
from os import listdir, makedirs
from os.path import exists, join, basename, isfile
import click


from model import *
from utils import *
import time

def test(image_path,weight_path,output_dir):
    g = generator_model()
    g.load_weights(weight_path)
    if not exists(output_dir):
        makedirs(output_dir)
    for f in listdir(image_path):
        if isfile(join(image_path,f)):
            image_dir=join(image_path,f)
            
            start= time.time()
            
            patches, dimW, dimH = load_test_patches(image_dir)
            
            
            generated_images = []
            for patch in zip(patches):
                x_test = np.array(preprocess_image(patch))
                
                generated = g.predict(x=x_test)

                generated_images.extend(deprocess_image(generated))
                
            input_image=reconstruct_image(patches,dimW,dimH)       
            generated_image=reconstruct_image(generated_images,dimW,dimH)
            
            end = time.time()
            
            print('Process_Time = {} seconds'.format(end-start))
            
            input_image=Image.fromarray(input_image.astype(np.uint8))
            generated_image = Image.fromarray(generated_image.astype(np.uint8))

            input_image.save(join(output_dir,'input_'+basename(image_dir))) 
            generated_image.save(join(output_dir,'generated_'+basename(image_dir)))  
            print('Output Saved as ' + output_dir + '/generated_'+basename(image_dir))


@click.command()
@click.option('--image_path', help='Image to deblur')
@click.option('--weight_path', help='Weights to apply')
@click.option('--output_dir', help='Output File Directory')
def test_command(image_path,weight_path,output_dir):
    return test(image_path,weight_path,output_dir)


if __name__ == "__main__":
    test_command()

