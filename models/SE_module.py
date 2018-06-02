import numpy as np
import pandas as pd
import random
import keras
from keras.optimizers import Adam
from keras.models import Input, Model
from keras.layers import Conv3D,BatchNormalization, Concatenate, MaxPooling3D, AveragePooling3D, UpSampling3D, Activation, Reshape, Permute, Add, Multiply
from keras.callbacks import EarlyStopping, ModelCheckpoint


def Squeeze_excitation_layer(input_x, out_dim, ratio):
    # TEST = keras.layer_global_average_pooling_3d(input_x)
    X1 = keras.layers.GlobalAveragePooling3D(data_format=None)(input_x)

    X1 = keras.layers.Dense(units=max(4, int(out_dim / ratio)), activation='relu', use_bias=True,
                            kernel_initializer='he_normal',
                            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(X1)

    X1 = keras.layers.Dense(units=out_dim, activation='sigmoid', use_bias=True,
                            kernel_initializer='he_normal',
                            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(X1)

    scale = Multiply()([input_x, X1])

    return scale
