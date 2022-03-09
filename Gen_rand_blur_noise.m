clc; clear;
rng default

BLUR=1; %% 0 No blur, 1 Lens Blur, 2 Gaussian Blur
NOISE=1; %% 0 No Noise, 1 Shot/Poisson, 2 AW-Gaussian
CS=128;  %% Image crop size (square)

Train_filedir='images\Train_HR'; %%Location of training images
Valid_filedir='images\Valid_HR'; %%Location of validation images
savedir='images'; %Location to save generated training datasets

n_files_train=50; %%Number of Training Images
n_files_valid=1; %%Number of Validation Images
n_files_valid_full=100; %%Number of Full size Validation Images

N_per_image=1; %%Nunber of crop per training image
N_per_image_valid=1; %%Nunber of crop per validation image

N_train_crops=n_files_train*N_per_image; %%Number of total training crops
N_valid_crops=n_files_valid*N_per_image_valid; %%Number of total validation crops

%Noise Parameters
Poisson_Noise_level=9;
Gaussian_Noise_variance=0.01;

%Blur Parameters
Xdim=25; %X dimension of Filter
Ydim=25; %Y dimension of Filter
Xsep=(1.55*10^-6); %Pixel Seperation in meters
Ysep=(1.55*10^-6); %Pixel Seperation
gauss_sigma=1.2;
I0=max(PSF_gauss(:));  %Peak amplitude of the disk
gamma=550*10^-9; %Wavelength of Light in meters
F=8; %F-number of Lens

var_lim=0.01; %%minimum Variation limit for crops



DIR_train=dir(fullfile(Train_filedir,'*.png'));
DIR_valid=dir(fullfile(Valid_filedir,'*.png'));

Input_folder=[savedir,'\Input'];
Input_valid_folder=[savedir,'\Input_valid'];
Input_valid_full_folder=[savedir,'\Input_valid_full'];
Target_folder=[savedir,'\Target'];
Target_valid_folder=[savedir,'\Target_valid'];
Target_valid_full_folder=[savedir,'\Target_valid_full'];
mkdir(savedir)
mkdir(Target_folder)
mkdir(Target_valid_folder)
mkdir(Target_valid_full_folder)
mkdir(Input_folder)
mkdir(Input_valid_folder)
mkdir(Input_valid_full_folder)

PSF_gauss=fspecial('gaussian',Xdim,gauss_sigma);
PSF_main=AiryFunction(Xdim,Ydim,Xsep,Ysep,I0,gamma,F);

figure(1)
subplot(1,2,1)
imagesc(PSF_gauss)
colormap(jet)
colorbar
title('Gaussian Blur Kernel')
subplot(1,2,2)
imagesc(PSF_main)
colormap(jet)
colorbar
title('Airy Disk Blur Kernel')

c=0;
index=randperm(n_files_train*N_per_image);
for i=1:n_files_train
    fid=DIR_train(i).name;
    img=imread(fullfile(Train_filedir,fid));
    W=size(img,1);
    H=size(img,2);
    
    for j=1:N_per_image
        c=c+1;
        
        X=ceil((W-(CS+Xdim))*rand(1));
        Y=ceil((H-(CS+Ydim))*rand(1));
       
        Targetpatch=imcrop(img,[Y,X,(CS+Xdim)-1,(CS+Ydim)-1]);
        
        while var(double(Targetpatch(:))/255) < var_lim
            X=ceil((W-(CS+Xdim))*rand(1));
            Y=ceil((H-(CS+Ydim))*rand(1));
            Targetpatch=imcrop(img,[Y,X,(CS+Xdim)-1,(CS+Ydim)-1]);
        end
        
                
        Inputpatch=double(Targetpatch);
        Inputpatch=Inputpatch/255;
        
        Targetpatch=imcrop(Targetpatch,[Ydim/2,Xdim/2,CS-1,CS-1]);
        imwrite(Targetpatch,[Target_folder,'/Scan_',num2str(index(c),'%05i'),'.png'])
        
        if BLUR == 1
            F_rand=F+((2*rand(1))-1);
            PSF=AiryFunction(Xdim,Ydim,Xsep,Ysep,I0,gamma,F);
            
            Blurpatch=double(Inputpatch);
            Blurpatch=imfilter(Blurpatch,PSF,'symmetric','conv');
            Blurpatch=Blurpatch/max(Blurpatch(:));
            
            Inputpatch=imcrop(Blurpatch,[Ydim/2,Xdim/2,CS-1,CS-1]);
        elseif Blur==2
            sigma_rand=gauss_sigma+(rand(1)*(gauss_sigma*0.1));
            kernel=fspecial('gaussian',Xdim,sigma_rand);
            
            Blurpatch=imfilter(Inputpatch,kernel,'symmetric','same','conv');
            Inputpatch=imcrop(Blurpatch,[Ydim/2,Xdim/2,CS-1,CS-1]);
        end
        
        if NOISE == 1
            %Add Poisson Noise
            NL=10^(Poisson_Noise_level+(rand(1)-0.5));
            Noisepatch=Inputpatch/(NL);
            Noisepatch=imnoise(Noisepatch,'Poisson');
            Noisepatch=Noisepatch*NL;
            
            Inputpatch=uint8(Noisepatch*255);
        elseif NOISE == 2
            %Add Gaussian Noise
            Noisepatch=imnoise(Inputpatch,'gaussian',0,Gaussian_Noise_variance+(((Gaussian_Noise_variance/10)*2*rand(1))-(Gaussian_Noise_variance/10)));
            Inputpatch=uint8(Noisepatch*255);
        end
        imwrite(Inputpatch,[Input_folder,'/Scan_',num2str(index(c),'%05i'),'.png'])
    end
end

c=0;
index=randperm(n_files_valid*N_per_image_valid);
for i=1:n_files_valid
    fid=DIR_valid(i).name;
    img=imread(fullfile(Valid_filedir,fid));
    W=size(img,1);
    H=size(img,2);
    
    for j=1:N_per_image_valid
        c=c+1;
        
        X=ceil((W-(CS+Xdim))*rand(1));
        Y=ceil((H-(CS+Ydim))*rand(1));
        
        
        Targetpatch=imcrop(img,[Y,X,(CS+Xdim)-1,(CS+Ydim)-1]);
        
        while var(double(Targetpatch(:))/255) < var_lim
            X=ceil((W-(CS+Xdim))*rand(1));
            Y=ceil((H-(CS+Ydim))*rand(1));
            Targetpatch=imcrop(img,[Y,X,(CS+Xdim)-1,(CS+Ydim)-1]);
        end
        
        
        Inputpatch=double(Targetpatch);
        Inputpatch=Inputpatch/255;
        
        
        Targetpatch=imcrop(Targetpatch,[Ydim/2,Xdim/2,CS-1,CS-1]);
        imwrite(Targetpatch,[Target_valid_folder,'/Scan_',num2str(index(c),'%05i'),'.png'])
        
        if BLUR == 1
            F_rand=F+((2*rand(1))-1);
            PSF=AiryFunction(Xdim,Ydim,Xsep,Ysep,I0,gamma,F);
            
            Blurpatch=double(Inputpatch);
            Blurpatch=imfilter(Blurpatch,PSF,'symmetric','conv');
            Blurpatch=Blurpatch/max(Blurpatch(:));
            
            Inputpatch=imcrop(Blurpatch,[Ydim/2,Xdim/2,CS-1,CS-1]);
        end
        
        if NOISE == 1
            %Add Poisson Noise
            NL=10^(Poisson_Noise_level+(rand(1)-0.5));
            Noisepatch=Inputpatch/(NL);
            Noisepatch=imnoise(Noisepatch,'Poisson');
            Noisepatch=Noisepatch*NL;
            
            Inputpatch=uint8(Noisepatch*255);
        elseif NOISE == 2
            %Add Gaussian Noise
            Noisepatch=imnoise(Inputpatch,'gaussian',0,Gaussian_Noise_variance+(((Gaussian_Noise_variance/10)*2*rand(1))-(Gaussian_Noise_variance/10)));
            Inputpatch=uint8(Noisepatch*255);
        end
        imwrite(Inputpatch,[Input_valid_folder,'/Scan_',num2str(index(c),'%05i'),'.png'])
        
    end
end

c=0;
CROP=0;
index=randperm(n_files_valid_full);
for i=1:n_files_valid_full
    fid=DIR_valid(i).name;
    img=imread(fullfile(Valid_filedir,fid));
    W=size(img,1);
    H=size(img,2);
    
    for j=1:1
        c=c+1;
        
        Targetpatch=img;
        
        Inputpatch=double(Targetpatch);
        Inputpatch=Inputpatch/255;
        
        imwrite(Targetpatch,[Target_valid_full_folder,'/Scan_',num2str(index(c),'%05i'),'.png'])
        
        if BLUR == 1
            F_rand=F+((2*rand(1))-1);
            PSF=AiryFunction(Xdim,Ydim,Xsep,Ysep,I0,gamma,F);
            
            Blurpatch=double(Inputpatch);
            Blurpatch=imfilter(Blurpatch,PSF,'symmetric','conv');
            Blurpatch=Blurpatch/max(Blurpatch(:));
            
            Inputpatch=Blurpatch;
        end
        
        if NOISE == 1
            %Add Poisson Noise
            NL=10^(Poisson_Noise_level+(rand(1)-0.5));
            Noisepatch=Inputpatch/(NL);
            Noisepatch=imnoise(Noisepatch,'Poisson');
            Noisepatch=Noisepatch*NL;
            
            Inputpatch=uint8(Noisepatch*255);
        elseif NOISE == 2
            %Add Gaussian Noise
            Noisepatch=imnoise(Inputpatch,'gaussian',0,Gaussian_Noise_variance+(((Gaussian_Noise_variance/10)*2*rand(1))-(Gaussian_Noise_variance/10)));
            Inputpatch=uint8(Noisepatch*255);
        end
        imwrite(Inputpatch,[Input_valid_full_folder,'/Scan_',num2str(index(c),'%05i'),'.png'])
        
    end
end

