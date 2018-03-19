import numpy as np
import pandas as pd
import random
import keras
from keras.optimizers import Adam
from keras.models import Input, Model
from keras.layers import Conv3D,BatchNormalization, Concatenate, MaxPooling3D, AveragePooling3D, UpSampling3D, Activation, Reshape, Permute
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras.layers.merge import add
from keras import backend as K

class ResNet50Model():
	# Params:
	# optimizer: keras optimizer (e.g. Adam)
	# loss: loss function for optimization
	# metrics: list of metrics (e.g. ['accuracy'])
	def __init__(self, optimizer, loss, metrics=['accuracy'], epochs=1, batch_size=16, model_name="unknown"):
		self.optimizer = optimizer
		self.model_name = model_name
		self.loss = loss
		self.metrics = metrics
		self.epochs = epochs
		self.batch_size = batch_size
		self.model = None


	def shortcut3d(self, input, residual):
		stride_dim1 = input._keras_shape[1] // residual._keras_shape[1]
		stride_dim2 = input._keras_shape[2] // residual._keras_shape[2]
		stride_dim3 = input._keras_shape[3] // residual._keras_shape[3]
		equal_channels = residual._keras_shape[4]  == input._keras_shape[4]

		shortcut = input
		if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
			shortcut = Conv3D(
		        filters=residual._keras_shape[4],
		        kernel_size=(1, 1, 1),
		        strides=(stride_dim1, stride_dim2, stride_dim3),
		        kernel_initializer="he_normal", padding="valid",
		        kernel_regularizer=l2(1e-4)
		        )(input)
		#if (shortcut.shape== residual.shape): return add([shortcut, residual])
		#return residual
		return add([shortcut, residual])


	def bottleneck(self, input, filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4), is_first_block_of_first_layer=False):
		if is_first_block_of_first_layer:
		    # don't repeat bn->relu since we just did bn->relu->maxpool
		    conv_1_1 = Conv3D(filters=filters, kernel_size=(1, 1, 1),
		                      strides=strides, padding="same",
		                      kernel_initializer="he_normal",
		                      kernel_regularizer=kernel_regularizer
		                      )(input)
		else:
		    conv_1_1 = self.batch_norm_relu(input)
		    conv_1_1 = Conv3D(filters=filters, kernel_size=(1, 1, 1),
		                               strides=strides,
		                               kernel_regularizer=kernel_regularizer
		                               )(conv_1_1)

		conv_3_3 = self.batch_norm_relu(conv_1_1)
		conv_3_3 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
		                           kernel_regularizer=kernel_regularizer
		                           )(conv_3_3)

		residual = self.batch_norm_relu(conv_3_3)
		residual = Conv3D(filters=filters * 4, kernel_size=(1, 1, 1),
		                           kernel_regularizer=kernel_regularizer
		                           )(residual)

		return self.shortcut3d(input, residual)


	def residual_block3d(self, input, filters, kernel_regularizer, repetitions, is_first_layer=False):
		for i in range(repetitions):
		    strides = (1, 1, 1)
		    if i == 0 and not is_first_layer:
		        strides = (2, 2, 2)
		    input = self.bottleneck(input, filters=filters, strides=strides,kernel_regularizer=kernel_regularizer,is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
		return input


	def batch_norm_relu(self, input):
		t = BatchNormalization(axis=4)(input)
		return Activation("relu")(t)


	# This function defines the baseline model in Keras
	def build_model(self, input_shape):
		reg_factor = 1e-4
		repetitions = [3, 4, 6, 3]
		# Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
		X_input = Input(shape = input_shape)
		
		D1 = Conv3D(filters=64, kernel_size=(7, 7, 7), strides=(2, 2, 2), kernel_regularizer=l2(reg_factor))(X_input)
		D1 = self.batch_norm_relu(D1)

		D2 =  MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(D1)


		D3 = self.residual_block3d(D2, filters=64,kernel_regularizer=l2(reg_factor),repetitions=repetitions[0], is_first_layer = True)(D2)
		D3 = self.residual_block3d(D3, filters=128,kernel_regularizer=l2(reg_factor),repetitions=repetitions[1], is_first_layer = False)(D3)
		D3 = self.residual_block3d(D3, filters=256,kernel_regularizer=l2(reg_factor),repetitions=repetitions[2], is_first_layer = False)(D3)
		D3 = self.residual_block3d(D3, filters=512,kernel_regularizer=l2(reg_factor),repetitions=repetitions[3], is_first_layer = False)(D3)

		#Axis could be 1
		D4 = self.batch_norm_relu(D3)

		D5 = AveragePooling3D(pool_size=(D3._keras_shape[1], D3._keras_shape[2], D3._keras_shape[3]),strides=(1,1,1), padding="same")(D4)
		D5 = Flatten()(D5)

		
		#pred = Dense(units=num_outputs, kernel_initializer="he_normal", activation="sigmoid", kernel_regularizer=l2(reg_factor))(D5)
	    
		pred = Conv3D(filters=1, kernel_size=1, activation='sigmoid')(D5)
		model = Model(inputs=X_input, outputs=pred, name='ResNet50')

		self.model = model

	def compile(self):
		self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
		self.model.summary()

	def fit(self, X_train, Y_train, validation_set=None, global_step=None):
		earlystopper = EarlyStopping(patience=10, verbose=1)
		checkpointer = ModelCheckpoint('model-' + self.model_name+ '-1.h5', verbose=1, save_best_only=True)
		if validation_set is None:
			return self.model.fit(x=X_train, y=Y_train, validation_split=0.2, epochs=self.epochs, batch_size=self.batch_size, callbacks=[earlystopper, checkpointer])
		else:
			if global_step is not None:
				return self.model.fit(x=X_train, y=Y_train, validation_data=validation_set, epochs=self.epochs, batch_size=self.batch_size, callbacks=[earlystopper, checkpointer], global_step=global_step)
			else:
				return self.model.fit(x=X_train, y=Y_train, validation_data=validation_set, epochs=self.epochs, batch_size=self.batch_size, callbacks=[earlystopper, checkpointer])

	def evaluate(self, X_test, Y_test):
		return self.model.evaluate(x=X_test, y=Y_test)


