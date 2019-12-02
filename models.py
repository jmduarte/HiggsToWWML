import keras
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Lambda, Activation
import keras.backend as K
from keras.backend import slice

def dense(ninputs, noutputs):

    inputs = Input(shape=(ninputs,), name = 'input')  
    x = BatchNormalization(name='bn_1')(inputs)
    x = Dense(64, name = 'dense_1', activation='relu')(inputs)
    x = Dropout(rate=0.1)(x)
    x = Dense(32, name = 'dense_2', activation='relu')(x)
    x = Dropout(rate=0.1)(x)
    x = Dense(32, name = 'dense_3', activation='relu')(x)
    x = Dropout(rate=0.1)(x)
    
    outputs = Dense(noutputs, name = 'output', activation='linear')(x)

    outputs1 = Lambda(lambda x: slice(x, (0, 0), (-1,  1)),name='lambda_1')(outputs)
    outputs2 = Lambda(lambda x: slice(x, (0, 1), (-1, -1)),name='lambda_2')(outputs)

    outputs1 = Activation('sigmoid',name='sigmoid_1')(outputs1)

    keras_model = Model(inputs=inputs, outputs=[outputs1, outputs2])

    return keras_model

