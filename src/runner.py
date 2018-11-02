
# coding: utf-8

# # 2nd milestone

# In[6]:


from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Flatten
import numpy as np
import scipy as sp
from skimage import color
import warnings
import keras
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


# In[3]:


import pandas as pd
import urllib.request
import uuid
from PIL import Image
import requests
from io import BytesIO
import glob

# Own .py files
import data_collector
import image_loader
import nnetwork

#import image_loader
image_folder = 'O:/ProgrammingSoftwares/anaconda_projects/dp_nagyhazi/samples/images/'
img_loader = image_loader.ImageLoader(image_folder)

# Separate_small_data(validation_rate, test_rate)
img_loader.separate_small_data(0.1,0.1)


# In[4]:


model = nnetwork.create_vgg_model()
model.compile('adam', loss = 'categorical_crossentropy',
              metrics=['accuracy', keras.metrics.categorical_accuracy])


# In[7]:


patience=20
early_stopping=EarlyStopping(patience=patience, verbose=1)
checkpointer=ModelCheckpoint(filepath='weights.hdf5', save_best_only=True, verbose=1)


# In[8]:


history = model.fit(x=img_loader.X_train,
                    y=img_loader.Y_train,
                    batch_size=32,
                    epochs=2,
                    validation_data=(img_loader.X_valid,img_loader.Y_valid),
                   callbacks=[checkpointer, early_stopping])


# In[9]:


score = model.evaluate(img_loader.X_test, img_loader.Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[12]:


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[17]:


from matplotlib.pyplot import imshow
from skimage import color
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

idx = 10

y_real = model.predict(img_loader.X_train[idx].reshape((1,224,224,3)))
y_real = np.apply_along_axis(lambda x: img_loader.pts_in_hull[np.argmax(x)], axis=3, arr = y_real)
img_loader.pts_in_hull # létező színosztályok

lab_im = np.concatenate([img_loader.y_dataset[np.newaxis,idx,:,:,0, np.newaxis],y_real ], axis=3)
rgb_im = color.lab2rgb(lab_im[0])
plt.imshow(rgb_im)


# In[ ]:


import cv2
import numpy as np

res = cv2.resize(img_loader.X_train[0], dsize=(56, 56), interpolation=cv2.INTER_CUBIC)
res = res.reshape((56,56,1))

