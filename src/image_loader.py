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



def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def create_onehot_vectors(node, nodes, onehot_list):
    idx = closest_node(node,nodes)
    return onehot_list[idx]

class ImageLoader:
    
    def __init__(self, folder, pt_in_hull_folder = 'O:\\ProgrammingSoftwares\\anaconda_projects\\dp_nagyhazi\\data\\pts_in_hull.npy'):
        
        self.pts_in_hull = np.load(pt_in_hull_folder)
        self.image_list = []
        
        for filename in glob.glob(folder + '*.jpg'): #
            img= np.asarray(Image.open(filename))
            img = img.astype(float)
            self.image_list.append(img)
            
    def separate_small_data(self,valid_split, test_split, input_size = 224, output_size = 56):
        self.dataset = []
        self.y_dataset = []
        for img_idx in range(len(self.image_list)):
           
            im = self.image_list[img_idx]            
            im = transform.resize( im , (224,224), preserve_range=True)
            im = im / 255.0
            #I'm needed the image in Lab color mode:
            lab_im = color.rgb2lab(im)
            #lab_im.dtype = float64
            lab_im = transform.resize( lab_im , (input_size,input_size), preserve_range=True)
            y_lab_im = transform.resize( lab_im , (output_size,output_size), preserve_range=True)
            #print(lab_im)
            self.dataset.append(np.asarray(lab_im, dtype="float"))
            self.y_dataset.append(np.asarray(y_lab_im, dtype="float"))
            
        #self.dataset = np.asarray(self.image_list, dtype="float32")
        self.dataset = np.array(self.dataset, dtype="float")
        self.y_dataset = np.array(self.y_dataset, dtype="float")
        
        X = np.concatenate([self.dataset[:,:,:,0, np.newaxis], self.dataset[:,:,:,0, np.newaxis], self.dataset[:,:,:,0, np.newaxis] ], axis = 3)
        #Normalize to 0...1 interval
        X = X / 100.0
        onehot_encoder = OneHotEncoder(sparse = False)
        #integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(np.array(range(0,self.pts_in_hull.shape[0])).reshape(-1,1) )
        #print(onehot_encoded)
        #self.pts_in_hull
        
        res = np.apply_along_axis(create_onehot_vectors, axis = 3, arr = self.y_dataset[:,:,:,1:], nodes = self.pts_in_hull, onehot_list = onehot_encoded)
        
        print(res.shape)
        
        Y = res
        
        v_index = int(X.shape[0] *  (1 - valid_split - test_split))
        t_index = int(X.shape[0] *  (1 - test_split))
        
        self.X_test = X[t_index:]
        self.Y_test = Y[t_index:]
        self.X_valid = X[v_index:t_index]
        self.Y_valid = Y[v_index:t_index]
        self.X_train = X[:v_index]
        self.Y_train = Y[:v_index]
        
        #print(X_test)
        #print(X_test.shape)
        #print(Y_test)
        #print(Y_test.shape)