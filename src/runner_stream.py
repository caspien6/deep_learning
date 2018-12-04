from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Flatten
import numpy as np
import scipy as sp
from skimage import color
import warnings
import keras
from keras.callbacks import EarlyStopping, LambdaCallback,ModelCheckpoint,CSVLogger

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
from nnetwork import weighted_categorical_crossentropy

from utility_methods import collect_and_separate_labels, collect_labels,save_plots, save_plots_callback
from streaming_data import StreamingDataGenerator
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

pts_hull_file = '/userhome/student/kede/colorize/deep_learning/data/pts_in_hull.npy'
distribution_file= '/userhome/student/kede/colorize/deep_learning/data/prior_probs.npy'

train_labl_path = '/userhome/student/kede/colorize/deep_learning/data/train_labels.csv'
valid_labl_path = '/userhome/student/kede/colorize/deep_learning/data/valid_labels.csv'
test_labl_path = '/userhome/student/kede/colorize/deep_learning/data/test_labels.csv'
class_desc_path = '/userhome/student/kede/colorize/deep_learning/data/class_descriptions.csv'
image_id_path = '/userhome/student/kede/colorize/deep_learning/data/image_ids_and_rotation.csv'

image_train_root_folder = '/userhome/student/kede/colorize/deep_learning/data/images/train_human/'
image_valid_root_folder = '/userhome/student/kede/colorize/deep_learning/data/images/valid_human/'
image_test_root_folder = '/userhome/student/kede/colorize/deep_learning/data/images/test/'

#data_hl = data_collector.DataCollector()
#data_hl.load_datas(image_id_path, train_labl_path, valid_labl_path, test_labl_path, class_desc_path)


#label_names = ['City', 'Skyline', 'Cityscape', 'Boathouse', 'Landscape lighting', 'Town square', 'College town', 'Town']
#label_names = ['Amusement park', 'Park', 'Skatepark', 'Highway', 'Bus garage', 'Portrait photography',
#'Portrait', 'Self-portrait', 'Crowd', 'Politician', 'Outdoor structure', 'Home door', 'Door']
#label_names = ['Crowd', 'Human face', 'Human', 'Red hair', 'Human hair color','Child', 'Music artist', 'Facial expression']
#collect_labels(data_hl, image_train_root_folder, label_names)

img_streamer_train = StreamingDataGenerator(image_train_root_folder, pt_in_hull_folder = pts_hull_file, batch_size=64, just_test = False)
img_streamer_valid = StreamingDataGenerator(image_valid_root_folder, pt_in_hull_folder = pts_hull_file, batch_size=64, just_test = False)
img_streamer_test = StreamingDataGenerator(image_test_root_folder, pt_in_hull_folder = pts_hull_file, batch_size=64, just_test = False)

distrib = np.load(distribution_file)

model = nnetwork.create_vgg_model(1,4)

los = weighted_categorical_crossentropy(distrib,lmb=0.5)
model.compile('adam', loss = los, metrics=['accuracy', keras.metrics.categorical_accuracy])

#model = keras.models.load_model('weights.hdf5')

patience=50
early_stopping=EarlyStopping(monitor='val_acc',patience=patience, verbose=1)
checkpointer=ModelCheckpoint(filepath='weights.hdf5',monitor='val_acc', save_best_only=True, verbose=1)

csv_logger = CSVLogger('training2.log', append=True)

history = model.fit_generator(generator=img_streamer_train,
                    validation_data=img_streamer_valid,
                    epochs=200,
                   callbacks=[csv_logger,checkpointer, early_stopping],
                   use_multiprocessing=False,
                    workers=6,
                    max_queue_size = 12)



score = model.evaluate_generator(img_streamer_test, verbose=1,
                    use_multiprocessing=False,
                    workers=8)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

save_plots(history)