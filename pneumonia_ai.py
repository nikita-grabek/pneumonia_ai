import keras
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers import Input
from keras.models import Model
from keras.regularizers import *


def get_model():
    aliases = {}
    Input_0 = Input(shape=(3, 1317, 1857), name='Input_0')
    convolution2d_254 = Convolution2D(name='convolution2d_254',nb_filter= 6,border_mode= 'same' ,nb_col= 3,nb_row= 3,dim_ordering= 'th' ,activation= 'relu' )(Input_0)
    maxpooling2d_49 = MaxPooling2D(name='maxpooling2d_49',strides= (2, 2),dim_ordering= 'th' )(convolution2d_254)
    convolution2d_256 = Convolution2D(name='convolution2d_256',nb_filter= 12,border_mode= 'same' ,nb_col= 2,nb_row= 2,dim_ordering= 'th' ,activation= 'relu' )(maxpooling2d_49)
    Dropout_15 = Dropout(name='Dropout_15',p= 0.2)(convolution2d_256)
    maxpooling2d_50 = MaxPooling2D(name='maxpooling2d_50',strides= (2, 2),dim_ordering= 'th' )(Dropout_15)
    convolution2d_258 = Convolution2D(name='convolution2d_258',nb_filter= 12,border_mode= 'same' ,nb_col= 2,nb_row= 2,dim_ordering= 'th' ,activation= 'relu' )(maxpooling2d_50)
    maxpooling2d_51 = MaxPooling2D(name='maxpooling2d_51',strides= (2, 2),dim_ordering= 'th' )(convolution2d_258)
    convolution2d_266 = Convolution2D(name='convolution2d_266',nb_filter= 16,border_mode= 'same' ,nb_col= 2,nb_row= 2,dim_ordering= 'th' ,activation= 'relu' )(maxpooling2d_51)
    maxpooling2d_53 = MaxPooling2D(name='maxpooling2d_53',strides= (2, 2),dim_ordering= 'th' )(convolution2d_266)
    convolution2d_270 = Convolution2D(name='convolution2d_270',nb_filter= 32,border_mode= 'same' ,nb_col= 2,nb_row= 2,dim_ordering= 'th' ,activation= 'relu' )(maxpooling2d_53)
    batchnormalization_206 = BatchNormalization(name='batchnormalization_206')(convolution2d_270)
    convolution2d_272 = Convolution2D(name='convolution2d_272',nb_filter= 32,border_mode= 'same' ,nb_col= 2,nb_row= 2,dim_ordering= 'th' ,activation= 'relu' )(batchnormalization_206)
    maxpooling2d_54 = MaxPooling2D(name='maxpooling2d_54',strides= (2, 2),dim_ordering= 'th' )(convolution2d_272)
    convolution2d_280 = Convolution2D(name='convolution2d_280',nb_filter= 32,border_mode= 'same' ,nb_col= 2,nb_row= 2,dim_ordering= 'th' ,activation= 'relu' )(maxpooling2d_54)
    maxpooling2d_56 = MaxPooling2D(name='maxpooling2d_56',strides= (2, 2),dim_ordering= 'th' )(convolution2d_280)
    Dropout_17 = Dropout(name='Dropout_17',p= 0.1)(maxpooling2d_56)
    convolution2d_285 = Convolution2D(name='convolution2d_285',nb_filter= 32,border_mode= 'same' ,nb_col= 2,nb_row= 2,dim_ordering= 'th' ,activation= 'relu' )(Dropout_17)
    maxpooling2d_57 = MaxPooling2D(name='maxpooling2d_57',strides= (2, 2),dim_ordering= 'th' )(convolution2d_285)
    flatten = Flatten(name='flatten')(maxpooling2d_57)
    dense_21 = Dense(name='dense_21',output_dim= 2048,activation= 'relu' )(flatten)
    dense_22 = Dense(name='dense_22',output_dim= 512,activation= 'linear' )(dense_21)
    batchnormalization_208 = BatchNormalization(name='batchnormalization_208')(dense_22)
    activation_109 = Activation(name='activation_109',activation= 'relu' )(batchnormalization_208)
    dropout_11 = Dropout(name='dropout_11',p= 0.1)(activation_109)
    dense_23 = Dense(name='dense_23',output_dim= 256,activation= 'linear' )(dropout_11)
    activation_110 = Activation(name='activation_110',activation= 'relu' )(dense_23)
    dense_24 = Dense(name='dense_24',output_dim= 2,activation= 'softmax' )(activation_110)

    model = Model([Input_0],[dense_24])
    return aliases, model


from keras.optimizers import *

def get_optimizer():
    return Adadelta()

def is_custom_loss_function():
    return False

def get_loss_function():
    return 'binary_crossentropy'

def get_batch_size():
    return 16

def get_num_epoch():
    return 30

def get_data_config():
    return '{"mapping": {"Bild": {"options": {"Width": "1317", "horizontal_flip": false, "height_shift_range": 0, "Normalization": true, "shear_range": 0, "Height": "1857", "Augmentation": false, "width_shift_range": 0, "rotation_range": 0, "Scaling": 1, "vertical_flip": false, "pretrained": "None", "Resize": true}, "shape": "", "type": "Image", "port": "InputPort0"}, "Ergebnis": {"options": {}, "shape": "", "type": "Categorical", "port": "OutputPort0"}}, "samples": {"validation": 1171, "training": 4684, "split": 1, "test": 0}, "dataset": {"samples": 5857, "type": "private", "name": "Lungenentzuendung"}, "datasetLoadOption": "batch", "kfold": 1, "shuffle": true, "numPorts": 1}'
