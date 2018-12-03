
from unet_zf import ZF_UNET_224
from keras.optimizers import Adam
from keras.layers import Conv2D
from keras.layers.core import Activation
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping, LambdaCallback,ModelCheckpoint, CSVLogger
import numpy as np

#own py
import streaming_unet_data
from utility_methods import collect_and_separate_labels, collect_labels,save_plots, save_plots_callback

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


train_labl_path = '/userhome/student/kede/colorize/deep_learning/data/train_labels.csv'
valid_labl_path = '/userhome/student/kede/colorize/deep_learning/data/valid_labels.csv'
test_labl_path = '/userhome/student/kede/colorize/deep_learning/data/test_labels.csv'
class_desc_path = '/userhome/student/kede/colorize/deep_learning/data/class_descriptions.csv'
image_id_path = '/userhome/student/kede/colorize/deep_learning/data/image_ids_and_rotation.csv'

image_train_root_folder = '/userhome/student/kede/colorize/deep_learning/data/images/train2/'
image_valid_root_folder = '/userhome/student/kede/colorize/deep_learning/data/images/valid/'
image_test_root_folder = '/userhome/student/kede/colorize/deep_learning/data/images/test/'
#data_hl = data_collector.DataCollector()
#data_hl.load_datas(image_id_path, train_labl_path, valid_labl_path, test_labl_path, class_desc_path)


#label_names = ['City', 'Skyline', 'Cityscape', 'Boathouse', 'Landscape lighting', 'Town square', 'College town', 'Town']
#collect_labels(data_hl, image_root_folder, label_names)

img_streamer_train = streaming_unet_data.StreamingUnet_DataGenerator(image_train_root_folder, batch_size=64, just_test = False)
img_streamer_valid = streaming_unet_data.StreamingUnet_DataGenerator(image_valid_root_folder, batch_size=32, just_test = False)
img_streamer_test = streaming_unet_data.StreamingUnet_DataGenerator(image_test_root_folder, batch_size=32, just_test = False)


model = ZF_UNET_224(weights='generator')
optim = Adam()
model.compile(optimizer=optim, loss='mse', metrics=['accuracy'])


model.layers.pop()
model.layers.pop()
model.layers.pop()
model.layers.pop()


conv_out = Conv2D(3, (1, 1), padding='same')(model.layers[-1].output)
o = Activation('tanh', name='loss')(conv_out)

model2 = Model(input=model.input, output=[o])


model2.save('starting_unet.h5')
model2 = load_model('starting_unet.h5')


for layer in model2.layers[:-7]:
    layer.trainable = False


model2.summary()
patience=30
early_stopping=EarlyStopping(monitor='loss',patience=patience, verbose=1)
checkpointer=ModelCheckpoint(filepath='unet_weights.hdf5', monitor='loss', save_best_only=True, verbose=1)
csv_logger = CSVLogger('unet_training.log', append=True)


model2.compile('adam', 'mse')

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
history = model2.fit_generator(generator=img_streamer_train,
                    validation_data=img_streamer_valid,
                    callbacks=[csv_logger,checkpointer, early_stopping],
                    epochs=200,
                   	use_multiprocessing=False,
                    workers=1,
                    verbose=1,
                    max_queue_size = 4)



score = model2.evaluate_generator(img_streamer_test, verbose=1,
                    use_multiprocessing=False,
                    workers=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

save_plots(history)
