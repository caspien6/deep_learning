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
from utility_methods import collect_and_separate_labels, collect_labels,save_plots, save_plots_callback
from streaming_data import StreamingDataGenerator
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

pts_hull_file = '/userhome/student/kede/colorize/deep_learning/data/pts_in_hull.npy'

train_labl_path = '/userhome/student/kede/colorize/deep_learning/data/train_labels.csv'
valid_labl_path = '/userhome/student/kede/colorize/deep_learning/data/valid_labels.csv'
test_labl_path = '/userhome/student/kede/colorize/deep_learning/data/test_labels.csv'
class_desc_path = '/userhome/student/kede/colorize/deep_learning/data/class_descriptions.csv'
image_id_path = '/userhome/student/kede/colorize/deep_learning/data/image_ids_and_rotation.csv'

image_train_root_folder = '/userhome/student/kede/colorize/deep_learning/data/images/train/'
image_valid_root_folder = '/userhome/student/kede/colorize/deep_learning/data/images/valid/'
image_test_root_folder = '/userhome/student/kede/colorize/deep_learning/data/images/test/'

#data_hl = data_collector.DataCollector()
#data_hl.load_datas(image_id_path, train_labl_path, valid_labl_path, test_labl_path, class_desc_path)


#label_names = ['City', 'Skyline', 'Cityscape', 'Boathouse', 'Landscape lighting', 'Town square', 'College town', 'Town']
#collect_labels(data_hl, image_root_folder, label_names)

img_streamer_train = StreamingDataGenerator(image_train_root_folder, pt_in_hull_folder = pts_hull_file, batch_size=32, just_test = False)
img_streamer_valid = StreamingDataGenerator(image_valid_root_folder, pt_in_hull_folder = pts_hull_file, batch_size=32, just_test = False)
img_streamer_test = StreamingDataGenerator(image_test_root_folder, pt_in_hull_folder = pts_hull_file, batch_size=32, just_test = False)

model = nnetwork.create_vgg_model(2)
model.compile('adam', loss = 'categorical_crossentropy', metrics=['accuracy', keras.metrics.categorical_accuracy])

patience=30
early_stopping=EarlyStopping(monitor='val_acc',patience=patience, verbose=1)
checkpointer=ModelCheckpoint(filepath='weights.hdf5',monitor='val_acc', save_best_only=True, verbose=1)

csv_logger = CSVLogger('training1.log', append=True)

history = model.fit_generator(generator=img_streamer_train,
                    validation_data=img_streamer_valid,
                    epochs=200,
                   callbacks=[csv_logger,checkpointer, early_stopping],
                   use_multiprocessing=True,
                    workers=1,
                    max_queue_size = 6)



score = model.evaluate_generator(img_streamer_test, verbose=1,
                    use_multiprocessing=False,
                    workers=8)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

save_plots(history)