import keras.losses as Kloss
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model
import numpy as np
import tensorflow as tf
import math
import scipy.io as io

# Note the image_shape must be multiple of patch_shape
image_size=128
image_shape = (image_size, image_size, 3)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)
    
def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

def MSE_loss(y_true, y_pred):
    return Kloss.mean_squared_error(y_pred , y_true)
    
def l1_loss(y_true, y_pred):
    return Kloss.mean_absolute_error(y_pred , y_true)

def VGG16_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

def VGG19_loss(y_true, y_pred):
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


def dark_channel_loss(y_true, y_pred):
    kernel_size=5
    kernel = [1,kernel_size, kernel_size, 1]
    
    min_channel_true = K.min(y_true, axis=3, keepdims=True)
    min_channel_pred = K.min(y_pred, axis=3, keepdims=True)
    
    dark_channel_image_true = -tf.nn.max_pool(-min_channel_true, ksize=kernel, strides=[1, 1, 1, 1],
                                         padding='SAME', name='erosion')
    dark_channel_image_pred = -tf.nn.max_pool(-min_channel_pred, ksize=kernel, strides=[1, 1, 1, 1],
                                         padding='SAME', name='erosion')
    
    dark_channel_loss = K.mean(K.square(dark_channel_image_true- dark_channel_image_pred))
    return dark_channel_loss
    
def gauss_kernel(error, sigma = 0.5):
    lambda1 = 1/(2*(sigma**2))
    G=(1/(np.sqrt(2*np.pi)*sigma))*K.exp(-lambda1*K.square(error))
    return G

def correntropy_025_loss( y_true, y_pred, sigma=0.25):
    error=y_true-y_pred
    G0=(1/(np.sqrt(2*np.pi)*sigma))
    Gxy=gauss_kernel(error,sigma)
    c_loss=G0-K.mean(Gxy)
    return c_loss

def correntropy_05_loss( y_true, y_pred, sigma=0.5):
    error=y_true-y_pred
    G0=(1/(np.sqrt(2*np.pi)*sigma))
    Gxy=gauss_kernel(error,sigma)
    c_loss=G0-K.mean(Gxy)
    return c_loss
    
def correntropy_075_loss( y_true, y_pred, sigma=0.75):
    error=y_true-y_pred
    G0=(1/(np.sqrt(2*np.pi)*sigma))
    Gxy=gauss_kernel(error,sigma)
    c_loss=G0-K.mean(Gxy)
    return c_loss

def l1_dark_channel_loss(y_true, y_pred,l1_lambda=1,dark_lambda=2.5):
    l1_cost=l1_lambda*l1_loss(y_true, y_pred)
    dark_cost=dark_lambda*dark_cost(y_true, y_pred)
    total_cost=l1_cost+dark_cost
    return total_cost

def l1_VGG16_loss(y_true, y_pred,VGG_lambda=1,l1_lambda=1):
    l1_cost=l1_lambda*l1_loss(y_true, y_pred)
    VGG_cost=VGG_lambda*VGG16_loss(y_true, y_pred)
    total_cost=l1_cost+VGG_cost
    return total_cost

def l1_VGG19_loss(y_true, y_pred,VGG_lambda=1,l1_lambda=1):
    l1_cost=l1_lambda*l1_loss(y_true, y_pred)
    VGG_cost=VGG_lambda*VGG19_loss(y_true, y_pred)
    total_cost=l1_cost+VGG_cost
    return total_cost

def VGG16_correntropy_05_loss(y_true, y_pred,VGG_lambda=1,correntropy_lambda=10):
    correntropy_cost=correntropy_lambda*correntropy_05_loss(y_true, y_pred)
    VGG_cost=VGG_lambda*VGG16_loss(y_true, y_pred)
    total_cost=correntropy_cost+VGG_cost
    return total_cost

def VGG16_correntropy_05_100_loss(y_true, y_pred,VGG_lambda=1,correntropy_lambda=100):
    correntropy_cost=correntropy_lambda*correntropy_05_loss(y_true, y_pred)
    VGG_cost=VGG_lambda*VGG16_loss(y_true, y_pred)
    total_cost=correntropy_cost+VGG_cost
    return total_cost