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
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def create_onehot_vectors(node, nodes, onehot_list):
    idx = closest_node(node,nodes)
    return onehot_list[idx]

class StreamingDataGenerator(keras.utils.Sequence):
    
    def __init__(self, folder, batch_size=32,
                 pt_in_hull_folder = 'O:\\ProgrammingSoftwares\\anaconda_projects\\dp_nagyhazi\\samples\\pts_in_hull.npy', just_test = False):
        
        'Initialization'
        self.batch_size = batch_size        
        self.pts_in_hull = np.load(pt_in_hull_folder)
        self.image_list = []
        for filename in glob.glob(folder + '*.jpg'):
            with Image.open(filename) as test_image:
                img= np.asarray(test_image)
                img = img.astype(float)
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
        gc.collect()
        'Returns augmented data with batch_size enzymes'
        input_size = 224
        output_size = 56
        # Initialization
        dataset = np.zeros((self.batch_size, input_size, input_size, 3))
        y_dataset = np.zeros((self.batch_size, output_size, output_size, 3))

        for img_idx in range(len(list_images_temp)):
            im = list_images_temp[img_idx]
            im = transform.resize( im , (input_size,input_size), preserve_range=True)
            im = im / 255.0
            #I'm needed the image in Lab color mode:
            if im.shape != (224,224,3):
                continue
            lab_im = color.rgb2lab(im)
            y_lab_im = transform.resize( lab_im , (output_size,output_size), preserve_range=True)

            dataset[img_idx] = np.asarray(lab_im, dtype="float")
            y_dataset[img_idx] = np.asarray(y_lab_im, dtype="float")
        
        X = np.concatenate([dataset[:,:,:,0, np.newaxis], dataset[:,:,:,0, np.newaxis], dataset[:,:,:,0, np.newaxis] ], axis = 3)

        #Normalize to 0...1 interval
        X = X / 100.0
        
        onehot_encoder = OneHotEncoder(sparse = False, categories='auto')
        onehot_encoded = onehot_encoder.fit_transform(np.array(range(0,self.pts_in_hull.shape[0])).reshape(-1,1) )

        #res = np.zeros((y_dataset.shape[0], y_dataset.shape[1], y_dataset.shape[2], self.pts_in_hull.shape[0]))

        #for idx in np.arange(y_dataset.shape[0]):
        #    for x in np.arange(y_dataset.shape[1]):
        #        for y in np.arange(y_dataset.shape[2]):
        #            res[idx,x,y] = create_onehot_vectors(node = y_dataset[idx,x,y,1:], nodes = self.pts_in_hull, onehot_list = onehot_encoded)
        
        res = np.apply_along_axis(create_onehot_vectors, axis = 3, arr = y_dataset[:,:,:,1:], nodes = self.pts_in_hull, onehot_list = onehot_encoded)
        Y = res
        return X, Y