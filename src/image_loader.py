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


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def create_onehot_vectors(node, nodes, onehot_list):
    idx = closest_node(node,nodes)
    return onehot_list[idx]

class ImageLoader:
    
    def __init__(self, folder, pt_in_hull_folder = 'O:\\ProgrammingSoftwares\\anaconda_projects\\dp_nagyhazi\\samples\\pts_in_hull.npy'):
        
        self.pts_in_hull = np.load(pt_in_hull_folder)
        self.image_list = []
        for filename in glob.glob(folder + '*.jpg'):
            with Image.open(filename) as test_image:
                img= np.asarray(test_image)
                img = img.astype(float)
                self.image_list.append(img)
        print('Length of image list: ',len(self.image_list))
            
    def separate_small_data(self,valid_split, test_split, input_size = 224, output_size = 56):
        self.dataset = np.zeros((len(self.image_list), input_size, input_size, 3))
        self.y_dataset = np.zeros((len(self.image_list), output_size, output_size, 3))

        for img_idx in range(len(self.image_list)):
            im = self.image_list[img_idx]
            im = transform.resize( im , (input_size,input_size), preserve_range=True)
            im = im / 255.0
            #I'm needed the image in Lab color mode:
            if im.shape != (224,224,3):
                continue
            lab_im = color.rgb2lab(im)
            #lab_im.dtype = float64
            #lab_im = transform.resize( lab_im , (input_size,input_size), preserve_range=True)
            y_lab_im = transform.resize( lab_im , (output_size,output_size), preserve_range=True)
            #print(lab_im)
            self.dataset[img_idx] = np.asarray(lab_im, dtype="float")
            self.y_dataset[img_idx] = np.asarray(y_lab_im, dtype="float")
        
        del self.image_list
        gc.collect()
        #self.dataset = np.asarray(self.image_list, dtype="float32")

        print('Gigabyte of dataset: ',float(self.dataset.nbytes)/ (1024**3) )
        print('Gigabyte of y_dataset: ',float(self.y_dataset.nbytes)/ (1024**3) )
        
        X = np.concatenate([self.dataset[:,:,:,0, np.newaxis], self.dataset[:,:,:,0, np.newaxis], self.dataset[:,:,:,0, np.newaxis] ], axis = 3)
        del self.dataset
        gc.collect()
        #Normalize to 0...1 interval
        X = X / 100.0
        onehot_encoder = OneHotEncoder(sparse = False)
        
        onehot_encoded = onehot_encoder.fit_transform(np.array(range(0,self.pts_in_hull.shape[0])).reshape(-1,1) )
        #print(onehot_encoded)
        #self.pts_in_hull
        res = np.zeros((self.y_dataset.shape[0], self.y_dataset.shape[1], self.y_dataset.shape[2], self.pts_in_hull.shape[0]))

        for idx in np.arange(self.y_dataset.shape[0]):
            for x in np.arange(self.y_dataset.shape[1]):
                for y in np.arange(self.y_dataset.shape[2]):
                    res[idx,x,y] = create_onehot_vectors(node = self.y_dataset[idx,x,y,1:], nodes = self.pts_in_hull, onehot_list = onehot_encoded)
        #res = np.apply_along_axis(create_onehot_vectors, axis = 3, arr = self.y_dataset[:,:,:,1:], nodes = self.pts_in_hull, onehot_list = onehot_encoded)
        
        
        Y = res
        
        v_index = int(X.shape[0] *  (1 - valid_split - test_split))
        t_index = int(X.shape[0] *  (1 - test_split))
        
        self.X_test = X[t_index:]
        self.Y_test = Y[t_index:]
        self.X_valid = X[v_index:t_index]
        self.Y_valid = Y[v_index:t_index]
        self.X_train = X[:v_index]
        self.Y_train = Y[:v_index]
        