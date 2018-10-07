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


class ImageLoader:
    
    def __init__(self, folder):
        #self.srgb_profile = ImageCms.createProfile("sRGB")
        #self.lab_profile  = ImageCms.createProfile("LAB")

        #self.rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(self.srgb_profile, self.lab_profile, "RGB", "LAB")
        
        self.image_list = []
        for filename in glob.glob(folder + '*.jpg'): #
            im = io.imread(filename)
            self.image_list.append(im)
            
    def separate_small_data(self,valid_split, test_split):
        self.dataset = []
        
        for img_idx in range(len(self.image_list)):
           
            im = self.image_list[img_idx]
            #I'm needed the image in Lab color mode:
            lab_im = color.rgb2lab(im)
            #lab_im.dtype = float64
            lab_im = transform.resize( lab_im , (300,300), preserve_range=True)         
            self.dataset.append(np.asarray(lab_im, dtype="float"))
            
        #self.dataset = np.asarray(self.image_list, dtype="float32")
        self.dataset = np.array(self.dataset, dtype="float")
        
        X = self.dataset[:,:,:,0]
        Y = self.dataset[:,:,:,1:]
        
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