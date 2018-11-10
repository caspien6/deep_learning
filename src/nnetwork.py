from keras.applications import VGG16
from keras import models
from keras.layers import BatchNormalization, Conv2D, UpSampling2D

def create_vgg_model(vgg_trainable_layer_count = 2, upsampling = 8):
    #Load the VGG model
    image_size = 224
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    # Freeze the layers except the last 4 layers
    for layer in vgg_conv.layers[:-vgg_trainable_layer_count]:
        layer.trainable = False

    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(vgg_conv)

    #conv8
    model.add(UpSampling2D(upsampling))
    model.add(Conv2D(313,3,padding = 'same', activation = 'relu'))
    model.add(Conv2D(313,1,padding = 'same', activation = 'softmax'))
    
    # Show a summary of the model. Check the number of trainable parameters
    model.summary()
    return model