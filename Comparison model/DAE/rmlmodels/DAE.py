import os

from keras.models import Model
from keras.layers import Input,Dense,Conv1D,MaxPool1D,ReLU,Dropout,Softmax,BatchNormalization,TimeDistributed
from keras.layers import CuDNNLSTM,Bidirectional,Flatten,LSTM,Reshape,Lambda
from keras.utils.vis_utils import plot_model

def DAE(weights=None,
             input_shape=[1024,2],
             classes=8,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    input = Input(input_shape,name='input')
    x = input
    dr=0
    #LSTM Unit
    x,s,c = LSTM(units=32,return_state = True,return_sequences = True)(x)
    x = Dropout(dr)(x)
    x,s1,c1 = LSTM(units=32,return_state = True,return_sequences = True)(x)
    #Classifier
    xc = Dense(32,activation='relu')(s1)
    xc = BatchNormalization()(xc)
    xc = Dropout(dr)(xc)
    xc = Dense(16,activation='relu')(xc)
    xc = BatchNormalization()(xc)
    xc = Dropout(dr)(xc)
    xc = Dense(classes,activation='softmax',name='xc')(xc)

    #Decoder
    xd = TimeDistributed(Dense(2),name='xd')(x)

    model = Model(inputs = input,outputs = [xc,xd])

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model