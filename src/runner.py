from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Flatten
import numpy as np
import scipy as sp
from skimage import color
import warnings
import keras
from keras.callbacks import EarlyStopping, LambdaCallback,ModelCheckpoint



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
from utility_methods import collect_and_separate_labels, collect_labels,save_plots, save_plots_callback


pts_hull_file = '/userhome/student/kede/colorize/deep_learning/data/pts_in_hull.npy'

train_labl_path = '/userhome/student/kede/colorize/deep_learning/data/train_labels.csv'
valid_labl_path = '/userhome/student/kede/colorize/deep_learning/data/valid_labels.csv'
test_labl_path = '/userhome/student/kede/colorize/deep_learning/data/test_labels.csv'
class_desc_path = '/userhome/student/kede/colorize/deep_learning/data/class_descriptions.csv'
image_id_path = '/userhome/student/kede/colorize/deep_learning/data/image_ids_and_rotation.csv'

image_root_folder = '/userhome/student/kede/colorize/deep_learning/data/images/'

#data_hl = data_collector.DataCollector()
#data_hl.load_datas(image_id_path, train_labl_path, valid_labl_path, test_labl_path, class_desc_path)


#label_names = ['City', 'Skyline', 'Cityscape', 'Boathouse', 'Landscape lighting', 'Town square', 'College town', 'Town']
#collect_labels(data_hl, image_root_folder, label_names)

img_loader = image_loader.ImageLoader(image_root_folder, pts_hull_file)

# Separate_small_data(validation_rate, test_rate)
img_loader.separate_small_data(0.2,0.1)


model = nnetwork.create_vgg_model(3)
model.compile('adam', loss = 'categorical_crossentropy', metrics=['accuracy', keras.metrics.categorical_accuracy])


patience=70
early_stopping=EarlyStopping(monitor='val_acc',patience=patience, verbose=1)
checkpointer=ModelCheckpoint(filepath='weights.hdf5',monitor='val_acc', save_best_only=True, verbose=1)
# Print the batch number at the beginning of every batch.
#epoch_plot_saver = LambdaCallback(on_epoch_end=lambda epoch,logs: save_plots_callback(logs) if epoch % 10 == 0 else False)

history = model.fit(x=img_loader.X_train,
                    y=img_loader.Y_train,
                    batch_size=64,
                    epochs=100,
                    validation_data=(img_loader.X_valid,img_loader.Y_valid),
                   callbacks=[checkpointer, early_stopping])



score = model.evaluate(img_loader.X_test, img_loader.Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

save_plots(history)