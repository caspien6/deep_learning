from keras.applications import VGG16
from keras import models, regularizers
from keras.layers import BatchNormalization, Conv2D, UpSampling2D
from keras.models import Model

def create_vgg_model(vgg_trainable_layer_count = 2, upsampling = 2):
    #Load the VGG model
    image_size = 224
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

    layer_name = 'block4_conv3'
    intermediate_layer_model = Model(inputs=vgg_conv.input,
                                     outputs=vgg_conv.get_layer(layer_name).output)
    for layer in intermediate_layer_model.layers[:-vgg_trainable_layer_count-1]:
        layer.trainable = False
    for layer in intermediate_layer_model.layers:
        print(layer.trainable)
    intermediate_layer_model.summary()
    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(intermediate_layer_model)

    #conv8
    model.add(UpSampling2D(upsampling))
    model.add(Conv2D(313,3,padding = 'same',kernel_regularizer=regularizers.l2(0.0001), activation = 'relu'))
    model.add(Conv2D(313,1,padding = 'same', activation = 'softmax'))
    
    # Show a summary of the model. Check the number of trainable parameters
    model.summary()
    return model