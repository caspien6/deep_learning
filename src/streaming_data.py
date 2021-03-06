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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import gc
import keras

def closest_node(node, nodes):
    """Helper function to get the discretized a,b pairs.
    """
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def create_onehot_vectors(node, nodes, onehot_list):
    idx = closest_node(node,nodes)
    return onehot_list[idx]

class StreamingDataGenerator(keras.utils.Sequence):
    """Multithread version custom data loader and generator for CLVGG models
    """
    def __init__(self, folder, batch_size=32,
                 pt_in_hull_folder = 'O:\\ProgrammingSoftwares\\anaconda_projects\\dp_nagyhazi\\samples\\pts_in_hull.npy', just_test = False,
                 random_trf = True):
        """It is loading up the images from the folder.
            #Arguments:
                folder: A folder path which contains .jpg images.

                batch_size: The size of the batches.

                pt_in_hull_folder: A filepath which refers to the pts_in_hull.npy file. 
                    It is containing 313 discretized a,b value pair.

                random_trf: Do you want random transformation when data loads into memory?
        """
        'Initialization'
        self.batch_size = batch_size        
        self.pts_in_hull = np.load(pt_in_hull_folder)
        self.image_list = []
        self.image_datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, zca_whitening= True, zca_epsilon= 0.1)

        for filename in glob.glob(folder + '*.jpg'):
            with Image.open(filename) as test_image:
                img= np.asarray(test_image)
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
        'Returns data with batch_size'
        gc.collect()
        input_size = 224
        output_size = 56
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

            rgb_im = transform.resize( im , (output_size,output_size), preserve_range=True)
            lab_im = color.rgb2lab(rgb_im/255)

            y_lab_im = lab_im*1.5
            y_dataset[img_idx] = np.asarray(y_lab_im, dtype="float")

        #if batch size is 1 then i want to display too.
        if (self.batch_size == 1):
            self.gray_img = dataset
        X = dataset - np.mean(dataset)
        
        onehot_encoder = OneHotEncoder(sparse = False, categories='auto')
        onehot_encoded = onehot_encoder.fit_transform(np.array(range(0,self.pts_in_hull.shape[0])).reshape(-1,1) )

        Y = np.apply_along_axis(create_onehot_vectors, axis = 3, arr = y_dataset[:,:,:,1:], nodes = self.pts_in_hull, onehot_list = onehot_encoded)
        return X, Y