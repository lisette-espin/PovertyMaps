import tensorflow as tf
import numpy as np
import random

# https://towardsdatascience.com/writing-a-custom-data-augmentation-layer-in-keras-2b53e048a98
# https://medium.com/featurepreneur/data-augmentation-using-keras-preprocessing-layers-6cdc7d49328e

class RandomColorDistortion(tf.keras.layers.Layer):
  
  def __init__(self, contrast_range=[0.5, 1.5],  brightness_delta=[-0.2, 0.2], **kwargs):
    super(RandomColorDistortion, self).__init__(**kwargs)
    self.contrast_range = contrast_range
    self.brightness_delta = brightness_delta
  
  def get_config(self):
        config = super().get_config()
        config.update({
            "contrast_range": self.contrast_range,
            "brightness_delta": self.brightness_delta,
        })
        return config
      
  def call(self, images, training=None):
    if not training:
      return images

    contrast = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
    brightness = np.random.uniform(self.brightness_delta[0], self.brightness_delta[1])
    images = tf.image.adjust_contrast(images, contrast)
    images = tf.image.adjust_brightness(images, brightness)
    images = tf.clip_by_value(images, 0, 1)
    return images

      
      
# #########################################################
# # CutMix
# ######################################################### 
# class CutMix(tf.keras.layers.Layer):
  
#   def __init__(self, dim, p, batch_size, **kwargs):
#     super(CutMix, self).__init__(**kwargs)
#     self.dim = dim
#     self.p = p
#     self.batch_size = batch_size
    
#   def call(self, images, training=None):
#     if not training:
#       return images

#     # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
#     # output - a batch of images with cutmix applied

#     imgs = []
#     for j in range(self.batch_size):
#       # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
#       P = tf.cast( tf.random.uniform([],0,1)<=self.p, tf.int32)

#       # CHOOSE RANDOM IMAGE TO CUTMIX WITH
#       k = tf.cast( tf.random.uniform([],0,self.batch_size),tf.int32)

#       # CHOOSE RANDOM LOCATION
#       x = tf.cast( tf.random.uniform([],0,self.dim),tf.int32)
#       y = tf.cast( tf.random.uniform([],0,self.dim),tf.int32)

#       b = tf.random.uniform([],0,1) # this is beta dist with alpha=1.0

#       WIDTH = tf.cast( self.dim * tf.math.sqrt(1-b),tf.int32) * P
#       ya = tf.math.maximum(0,y-WIDTH//2)
#       yb = tf.math.minimum(self.dim,y+WIDTH//2)
#       xa = tf.math.maximum(0,x-WIDTH//2)
#       xb = tf.math.minimum(self.dim,x+WIDTH//2)

#       # MAKE CUTMIX IMAGE
#       one = images[j,ya:yb,0:xa,:]
#       two = images[k,ya:yb,xa:xb,:]
#       three = images[j,ya:yb,xb:self.dim,:]
#       middle = tf.concat([one,two,three],axis=1)
#       img = tf.concat([images[j,0:ya,:,:],middle,images[j,yb:self.dim,:,:]],axis=0)
#       imgs.append(img)

#     # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
#     image2 = tf.reshape(tf.stack(imgs),(self.batch_size,self.dim,self.dim,3))
   
#     return image2
    
# #########################################################
# # MixUp
# #########################################################
# class MixUp(tf.keras.layers.Layer):
  
#   def __init__(self, dim, p, batch_size, **kwargs):
#     super(MixUp, self).__init__(**kwargs)
#     self.dim = dim
#     self.p = p
#     self.batch_size = batch_size
    
#   def call(self, images, training=None):
#     if not training:
#       return images
    
#     imgs = []
#     for j in range(self.batch_size):
#       # DO MIXUP WITH PROBABILITY DEFINED ABOVE
#       P = tf.cast( tf.random.uniform([],0,1)<=self.p, tf.float32)

#       # CHOOSE RANDOM
#       k = tf.cast( tf.random.uniform([],0,self.batch_size),tf.int32)
#       a = tf.random.uniform([],0,1)*P # this is beta dist with alpha=1.0

#       # MAKE MIXUP IMAGE
#       img1 = images[j,]
#       img2 = images[k,]
#       imgs.append((1-a)*img1 + a*img2)

#     # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
#     image2 = tf.reshape(tf.stack(imgs),(self.batch_size,self.dim,self.dim,3))
#     return image2

  
# #########################################################
# # MosaicMix
# #########################################################
# class MosaicMix(tf.keras.layers.Layer):
  
#   def __init__(self, dim, minfrac=0.25, maxfrac=0.75, **kwargs):
#     super(MosaicMix, self).__init__(**kwargs)
#     self.dim = dim
#     self.minfrac = minfrac
#     self.maxfrac = maxfrac
    
#   def call(self, images, training=None):
#     if not training:
#       return images
    
#     xc, yc  = np.random.randint(self.dim * self.minfrac, self.dim * self.maxfrac, (2,))
#     indices = np.random.permutation(int(images.shape[0]))
#     mosaic_image = np.zeros((self.dim, self.dim, 3), dtype=np.float32)
#     final_imgs   = []

#     # Iterate over the full indices 
#     for j in range(len(indices)): 
#       # Take 4 sample for to create a mosaic sample randomly 
#       rand4indices = [j] + random.sample(list(indices), 3) 

#       # Make mosaic with 4 samples 
#       for i in range(len(rand4indices)):
#         if i == 0:    # top left
#           x1a, y1a, x2a, y2a =  0,  0, xc, yc
#           x1b, y1b, x2b, y2b = self.dim - xc, self.dim - yc, self.dim, self.dim # from bottom right        
#         elif i == 1:  # top right
#           x1a, y1a, x2a, y2a = xc, 0, self.dim , yc
#           x1b, y1b, x2b, y2b = 0, self.dim - yc, self.dim - xc, self.dim # from bottom left
#         elif i == 2:  # bottom left
#           x1a, y1a, x2a, y2a = 0, yc, xc, self.dim
#           x1b, y1b, x2b, y2b = self.dim - xc, 0, self.dim, self.dim-yc   # from top right
#         elif i == 3:  # bottom right
#           x1a, y1a, x2a, y2a = xc, yc,  self.dim, self.dim
#           x1b, y1b, x2b, y2b = 0, 0, self.dim-xc, self.dim-yc    # from top left

#         # Copy-Paste
#         print('y1a',y1a)
#         print('y2a',y2a)
#         print('x1a',x1a)
#         print('x2a',x2a)
        
#         print('y1b',y1b)
#         print('y2b',y2b)
#         print('x1b',x1b)
#         print('x2b',x2b)
        
#         print(i)
#         print('images.shape[0]', images.shape[0])
        
#         try:
#           print('mosaic',mosaic_image[y1a:y2a, x1a:x2a].shape)
#         except Exception as ex:
#           print(ex)
          
#         try:
#           print('images',images[i][y1b:y2b, x1b:x2b].shape)
#         except Exception as ex:
#           print(ex)
          
        
#         mosaic_image[y1a:y2a, x1a:x2a] = images[i][y1b:y2b, x1b:x2b]

#       # Append the Mosiac samples
#       final_imgs.append(mosaic_image)

#     return final_imgs
    

# #########################################################
# # FMix
# #########################################################
# from utils.fmix_utils import sample_mask
# class FMix(tf.keras.layers.Layer):
  
#   def __init__(self, dim, alpha=1, decay_power=3, max_soft=0.0, reformulate=False, **kwargs):
#     super(FMix, self).__init__(**kwargs)
#     self.dim = dim
#     self.alpha = alpha
#     self.decay_power = decay_power
#     self.max_soft = max_soft
#     self.reformulate = reformulate
    
#   def call(self, images, training=None):
#     if not training:
#       return images
    
#     lam, mask = sample_mask(self.alpha, self.decay_power,(self.dim, self.dim), self.max_soft, self.reformulate)
#     index = tf.constant(np.random.permutation(int(images.shape[0])))
#     mask  = np.expand_dims(mask, -1)

#     # samples 
#     image1 = images * mask
#     image2 = tf.gather(images, index) * (1 - mask)
#     image3 = image1 + image2

#     return image3
