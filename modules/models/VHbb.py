import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from operator import *
from itertools import *
from keras.layers.normalization import BatchNormalization
from multi_gpu_model import multi_gpu_model


kernel_initializer = 'he_normal'
kernel_initializer_fc = 'lecun_uniform'



def FC(data, num_hidden, act='relu', p=None, name=''):
    if act=='leakyrelu':
        fc = keras.layers.Dense(num_hidden, activation='linear', name='%s_%s' % (name,act), kernel_initializer=kernel_initializer_fc)(data) # Add any layer, with the default of a linear squashing function                                                                                                                                                                                                           
        fc = keras.layers.advanced_activations.LeakyReLU(alpha=.001)(fc)   # add an advanced activation                                                                                                     
    else:
        fc = keras.layers.Dense(num_hidden, activation=act, name='%s_%s' % (name,act), kernel_initializer=kernel_initializer_fc)(data)
    if not p:
        return fc
    else:
        dropout = keras.layers.Dropout(rate=p, name='%s_dropout' % name)(fc)
        return dropout

#def VHbb(inputs, num_classes,num_regclasses, **kwargs):
def VHbb(inputs, num_classes, num_regclasses, datasets, multi_gpu=1, **kwargs): 
    input_db = inputs[0]

    print input_db.shape
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = keras.layers.Flatten()(input_db)
    print x.shape
    
    
    fc = FC(x,  20,  p=0.05, name='fc1')
    fc = FC(fc, 20,  p=0.05, name='fc2')
    fc = FC(fc, 6,  p=0.05, name='fc3')
    #fc = FC(x,  128,  name='fc1')
    #fc = FC(fc, 128, p=0.05,  name='fc2')
    #fc = FC(fc, 128, p=0.05, name='fc4')
    #fc = FC(fc, 64,  p=0.05, name='fc5')
    output = keras.layers.Dense(num_classes, activation='sigmoid', name='sigmoid', kernel_initializer=kernel_initializer_fc)(fc)

    print output.shape
    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model


def VHbb_tmva(inputs, num_classes, num_regclasses, datasets, multi_gpu=1, **kwargs):
    
    input_db = inputs[0]

    print input_db.shape
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = keras.layers.Flatten()(input_db)
    print x.shape
    
    
    fc = FC(x,  40, p=0.65, name='fc1')
    fc = FC(fc, 40, p=0.65,  name='fc2')
    fc = FC(fc, 40,  name='fc3')
    output = keras.layers.Dense(num_classes, activation='sigmoid', name='sigmoid', kernel_initializer=kernel_initializer_fc)(fc)

    print output.shape
    model = keras.models.Model(inputs=inputs, outputs=output)

    print model.summary()
    return model

