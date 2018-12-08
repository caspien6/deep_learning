import pandas as pd
import urllib.request
import uuid
from PIL import Image
import requests
from io import BytesIO
import glob
import numpy as np

from PIL import Image, ImageCms
from skimage import io, color,transform, img_as_float
from skimage.viewer import ImageViewer
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.preprocessing import StandardScaler
import gc
import keras

class StreamingUnet_DataGenerator(keras.utils.Sequence):
    """Multithread version custom data loader and generator for Unet models
    """
    
    def __init__(self, folder, batch_size=32, just_test = False, random_trf = False):
        """It is loading up the images from the folder.
            #Arguments:
                folder: A folder path which contains .jpg images.

                batch_size: The size of the batches.

                just_test: If you want to only test, 
                then just one batch of images will be load into the memory.

                random_trf: Do you want random transformation when data loads into memory?
        """
        'Initialization'
        self.batch_size = batch_size        
        self.image_list = []
        self.image_datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, zca_whitening= True, zca_epsilon= 0.1)

        for filename in glob.glob(folder + '*.jpg'):
            with Image.open(filename) as test_image:
                img = np.asarray(test_image)
                img = img.astype(float)
                
                if random_trf:
                    img = self.image_datagen.random_transform(img, seed=21)
                
                self.image_list.append(img)
            if just_test and len(self.image_list) > batch_size:
                break
        print('Length of image list: ',len(self.image_list))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = [k for k in range(index*self.batch_size,(index+1)*self.batch_size,1)]

        # Find list of IDs unicode(x.strip()) if x is not None else ''
        list_images_temp = [self.image_list[k] if len(self.image_list) > k else self.image_list[k - len(self.image_list)]  for k in indexes]

        # Generate data
        X, y = self.__data_augmentation(list_images_temp)

        return X, y
    
    def __data_augmentation(self, list_images_temp):
        """Arguments: 
            list_images_temp: List containing image indexes from self.image_list
        """
        gc.collect()
        'Returns augmented data with batch_size'
        input_size = 224
        output_size = 224
        
        # Initialization
        dataset = np.zeros((self.batch_size, input_size, input_size, 3))
        y_dataset = np.zeros((self.batch_size, output_size, output_size, 3))

        for img_idx in range(len(list_images_temp)):
            im = list_images_temp[img_idx]
            im = transform.resize( im , (input_size,input_size), preserve_range=True)
            gray_im = color.rgb2gray(im)
            dataset[img_idx] = np.concatenate([gray_im[:,:,np.newaxis], gray_im[:,:,np.newaxis], gray_im[:,:,np.newaxis] ], axis = 2 )

            
            #I'm needed the image in Lab color mode:
            if im.shape != (224,224,3):
                continue

            y_lab_im = color.rgb2lab(im/255)
            self.random = y_lab_im
            scalers = {}
            for i in range(y_dataset.shape[3]):
                scalers[i] = StandardScaler()
                y_lab_im[:,:,i] = ((y_lab_im[:,:,i] - np.min(y_lab_im[:,:,i])) / (np.max(y_lab_im[:,:,i]) - np.min(y_lab_im[:,:,i]))-0.5)*2
                y_dataset[img_idx,:,:,i] = np.asarray(y_lab_im[:,:,i], dtype="float")
            

        #if batch size is 1 then i want to display too.
        if (self.batch_size == 1):
            self.gray_img = dataset
        X = dataset
        Y = y_dataset[:,:,:,1:]
        return X, Y