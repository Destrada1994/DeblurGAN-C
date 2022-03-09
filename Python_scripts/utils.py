import os
from PIL import Image
import math
import numpy as np
import tensorflow as tf
from scipy import signal, ndimage
import random

Scale=1
Image_Size=128
OL=32
RESHAPE = (Image_Size,Image_Size)

def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False

def list_image_files(directory):
    files = sorted(os.listdir(directory))
    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]

def load_image(Input_path,Target_path):
    img1 = Image.open(Input_path)
    img2 = Image.open(Target_path)
    
    width, height = img1.size   # Get dimensions
    
    #Center Crop
    if width > Image_Size:
        left = math.floor((width - Image_Size)/2)
        right = left+Image_Size
    else:
        left=0
        right=width
    if height > Image_Size:
        top = math.floor((height - Image_Size)/2)
        bottom = top+Image_Size
    else:
        top=0;
        bottom=height;
    
    img1 = img1.crop((left, top, right, bottom)) #Crop Image 1
    img2 = img2.crop((left, top, right, bottom)) #Crop Image 2
    
    return img1, img2

def preprocess_image(cv_img):
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5

    return img


def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')


def load_images(path, input_folder,n_images_train,n_images_valid):
    if n_images_train < 0:
        n_images_train = float("inf")
    if n_images_valid < 0:
        n_images_valid = float("inf")
    print('Loading Images...')
    A_paths, B_paths = os.path.join(path, input_folder), os.path.join(path, 'Target')
    C_paths, D_paths = os.path.join(path, input_folder+'_valid'), os.path.join(path, 'Target_valid')
    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
    all_C_paths, all_D_paths = list_image_files(C_paths), list_image_files(D_paths)
    images_A, images_B = [], []
    images_C, images_D = [], []
    images_A_paths, images_B_paths = [], []
    images_C_paths, images_D_paths = [], []
    for path_A, path_B in zip(all_A_paths, all_B_paths):
        img_A, img_B = load_image(path_A, path_B)
        images_A.append(preprocess_image(img_A))
        images_B.append(preprocess_image(img_B))
        images_A_paths.append(path_A)
        images_B_paths.append(path_B)
        if len(images_A) > n_images_train - 1: break
        
    for path_C, path_D in zip(all_C_paths, all_D_paths):
        img_C, img_D = load_image(path_C, path_D)
        images_C.append(preprocess_image(img_C))
        images_D.append(preprocess_image(img_D))
        images_C_paths.append(path_C)
        images_D_paths.append(path_D)
        if len(images_C) > n_images_valid - 1: break
        
    print('{} Images Loaded'.format(len(images_A)))
    return {
        'A': np.array(images_A),
        'A_paths': np.array(images_A_paths),
        'B': np.array(images_B),
        'B_paths': np.array(images_B_paths),
        'C': np.array(images_C),
        'C_paths': np.array(images_C_paths),
        'D': np.array(images_D),
        'D_paths': np.array(images_D_paths)
    }

def write_log(callback, names, logs, batch_no):
    """
    Util to write callback for Keras training
    """
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def load_test_patches_2(path):
    img = Image.open(path)
    
    width, height = img.size   # Get dimensions
    
    dimW=int(Image_Size+(math.floor((width-Image_Size)/(Image_Size-OL))*(Image_Size-OL)))
    dimH=int(Image_Size+(math.floor((height-Image_Size)/(Image_Size-OL))*(Image_Size-OL)))
    
    if dimH!=height:
        Top = math.floor((height-dimH)/2)
        Bottom = Top+dimH
    else:
        Top=0
        Bottom=Top+dimH
    if dimW!=width:
        Left = math.floor((width-dimW)/2)
        Right = Left+dimW
    else:
        Left=0
        Right=Left+dimW
    
    img = img.crop((Left, Top, Right, Bottom)) #Crop Image
    
    i_width=int((dimW-Image_Size)/(Image_Size-OL)+1)
    i_height=int((dimH-Image_Size)/(Image_Size-OL)+1)

    patches=[]
    for i in range(i_height):
        if i==0:
            T=0
        else:
            T=T+(Image_Size-OL)
            
        for j in range(i_width):
            if j==0:
                L=0
            else:
                L=L+(Image_Size-OL)
                
            box=(L,T,L+Image_Size,T+Image_Size)
            
            patch=img.crop(box)
            
            patches.append(np.array(patch))
            
    return patches, dimW, dimH

def reconstruct_image_2(img_list, dimW, dimH):

    i_width=int((dimW-Image_Size)/(Image_Size-OL)+1)
    i_height=int((dimH-Image_Size)/(Image_Size-OL)+1)

    H=1
    V=1
    
    for patch in img_list:
        img=Image.fromarray(patch)
        if H==1:
            box2=np.array(img.crop((0,0,(Image_Size-OL),Image_Size)))
            box3=np.array(img.crop(((Image_Size-OL),0,(Image_Size-(OL/2)),Image_Size)))
            
            prev_box3=box3
            LINE=box2
            H=H+1
            
        elif H==i_width:
            box1=np.array(img.crop((OL/2,0,OL,Image_Size)))
            box2=np.array(img.crop((OL,0,Image_Size,Image_Size)))
            
            AVG=np.concatenate((prev_box3,box1),axis=1)
            CROP=np.concatenate((AVG,box2),axis=1)
            LINE=np.concatenate((LINE,CROP), axis=1)
            
            LINE=Image.fromarray(LINE.astype('uint8'))
            if V==1:
                box5=np.array(LINE.crop((0,0,dimW,(Image_Size-OL))))
                box6=np.array(LINE.crop((0,(Image_Size-OL),dimW,(Image_Size-(OL/2)))))
                
                Full=box5
                prev_box6=box6
                V=V+1
            elif V==i_height:
                box4=np.array(LINE.crop((0,OL/2,dimW,OL)))
                box5=np.array(LINE.crop((0,OL,dimW,Image_Size)))
                
                AVG2=np.concatenate((prev_box6,box4), axis=0)
                CROP2=np.concatenate((AVG2,box5), axis=0)
                Full=np.concatenate((Full,CROP2), axis=0)
                break
            else:
                box4=np.array(LINE.crop((0,OL/2,dimW,OL)))
                box5=np.array(LINE.crop((0,OL,dimW,(Image_Size-OL))))
                box6=np.array(LINE.crop((0,(Image_Size-OL),dimW,(Image_Size-(OL/2)))))
                
                AVG2=np.concatenate((prev_box6,box4), axis=0)
                prev_box6=box6
                CROP2=np.concatenate((AVG2,box5), axis=0)
                Full=np.concatenate((Full,CROP2), axis=0)
                V=V+1
            H=1
        else:
            box1=np.array(img.crop((OL/2,0,OL,Image_Size)))
            box2=np.array(img.crop((OL,0,(Image_Size-OL),Image_Size)))
            box3=np.array(img.crop(((Image_Size-OL),0,(Image_Size-(OL/2)),Image_Size)))
            
            AVG=np.concatenate((prev_box3,box1),axis=1)
            prev_box3=box3
            CROP=np.concatenate((AVG,box2),axis=1)
            LINE=np.concatenate((LINE,CROP), axis=1)
            H=H+1
    
    return Full.astype('uint8')

def load_test_patches(path):
    img = Image.open(path)
    
    width, height = img.size   # Get dimensions
    
    AR=Image_Size-OL
    
    dimW=math.floor(width/AR)*AR
    dimH=math.floor(height/AR)*AR
    if dimH!=height:
        Top = math.floor((height-dimH)/2)
        Bottom = Top+dimH
    else:
        Top=0
        Bottom=Top+dimH
    if dimW!=width:
        Left = math.floor((width-dimW)/2)
        Right = Left+dimW
    else:
        Left=0
        Right=Left+dimW
    
    img = img.crop((Left, Top, Right, Bottom)) #Crop Image
    

    patches=[]
    for i in range(0,dimH,AR):
        for j in range(0,dimW,AR):
            if j==0:
                L=0
            elif j==dimW-AR:
                L=j-OL
            else:
                L=j-(OL/2)
            if i==0:
                T=0
            elif i==dimH-AR:
                T=i-OL
            else:
                T=i-(OL/2)
        
            box=(L,T,L+Image_Size,T+Image_Size)
            
            patch=img.crop(box)
            
            patches.append(np.array(patch))
            
    return patches, dimW, dimH

def reconstruct_image(img_list, dimW, dimH):
    
    AR=(Image_Size-OL)
    NPH=(dimW)/AR
    NPV=(dimH)/AR
    H=1
    V=1
    
    overlap=[]
    for patch in img_list:
        img=Image.fromarray(patch)
        if H==1:
            L2=0
            L3=L2+AR
        elif H==NPH:
            L1=OL/2
            L2=L1+OL/2
        else:
            L1=0
            L2=L1+OL/2
            L3=L2+AR
        
        box1=np.array(img.crop((L1,T1,L1+AR,T1+AR)))
        RA=box1
        if H==1:
            A=RA
            H=H+1
        elif H>1 and H!=NPH:
            A=np.concatenate((A,RA), axis=1)
            H=H+1
        elif H==NPH:
            A=np.concatenate((A,RA), axis=1)
            H=1
            if V==1:
                Full=A
                V=V+1
            else:
                Full=np.concatenate((Full,A), axis=0)
                V=V+1
                
    return Full