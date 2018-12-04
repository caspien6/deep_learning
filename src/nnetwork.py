from keras.applications import VGG16
from keras import models, regularizers
from keras.layers import BatchNormalization, Conv2D, UpSampling2D
from keras.models import Model

from keras import backend as K
def weighted_categorical_crossentropy(distributions, lmb=0.5):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    lmb = K.variable(lmb)
    distributions = K.variable(distributions, dtype='float32')
    def my_elementwise_func(x):
        x = K.cast(x,'int64')
        return distributions[x]
        
    def recursive_map(inputs):
        if K.ndim(inputs) > 0:
            return K.map_fn(recursive_map, inputs, 'go_into')
        else:
            return my_elementwise_func(inputs)
    

    def loss(y_true, y_pred):
        
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        belso = K.sum(y_true * K.log(y_pred), axis=-1, keepdims=False)
        
        max_indices = K.argmax(y_true, axis=-1)
        max_indices = K.cast(max_indices,'float32')
        p_klp = recursive_map(max_indices)
        weights = 1/((1-lmb)*p_klp+ (lmb/2))
        los = K.sum(belso, -1)
        los = K.sum(los, -1)
        return -los
    
    return loss


def create_vgg_model(vgg_trainable_layer_count = 1, upsampling = 4):
    #Load the VGG model
    image_size = 224
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

    layer_name = 'block5_pool'
    intermediate_layer_model = Model(inputs=vgg_conv.input,
                                     outputs=vgg_conv.get_layer(layer_name).output)
    for layer in intermediate_layer_model.layers[:-vgg_trainable_layer_count]:
        layer.trainable = False
    for layer in intermediate_layer_model.layers:
        print(layer.trainable)
    intermediate_layer_model.summary()
    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(intermediate_layer_model)
    model.add(UpSampling2D(round(upsampling/2)))
    model.add(Conv2D(256,3,padding = 'same', activation = 'relu'))
    model.add(Conv2D(256,3,padding = 'same', activation = 'relu'))
    #conv8
    model.add(UpSampling2D(round(upsampling)))
    model.add(Conv2D(256,3,padding = 'same', activation = 'relu'))
    model.add(Conv2D(313,3,padding = 'same', activation = 'relu'))
    model.add(Conv2D(313,3,padding = 'same', activation = 'relu'))
    model.add(Conv2D(313,1,padding = 'same', activation = 'softmax'))
    
    # Show a summary of the model. Check the number of trainable parameters
    model.summary()
    return model