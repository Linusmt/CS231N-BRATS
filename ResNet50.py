import numpy as np
import pandas as pd
import random
import keras
from keras.optimizers import Adam
from keras.models import Input, Model
from keras.layers import Conv3D,BatchNormalization, Concatenate, MaxPooling3D, AveragePooling3D, UpSampling3D, Activation, Reshape, Permute
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2

class ResNet50Model():
	# Params:
	# optimizer: keras optimizer (e.g. Adam)
	# loss: loss function for optimization
	# metrics: list of metrics (e.g. ['accuracy'])
	def __init__(self, optimizer, loss, metrics=['accuracy'], epochs=1, batch_size=16):
		self.optimizer = optimizer
		self.loss = loss
		self.metrics = metrics
		self.epochs = epochs
		self.batch_size = batch_size
		self.model = None



	def bottleneck(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4), is_first_block_of_first_layer=False):
		if is_first_block_of_first_layer:
		    # don't repeat bn->relu since we just did bn->relu->maxpool
		    conv_1_1 = Conv3D(filters=filters, kernel_size=(1, 1, 1),
		                      strides=strides, padding="same",
		                      kernel_initializer="he_normal",
		                      kernel_regularizer=kernel_regularizer
		                      )(input)
		else:
		    conv_1_1 = _bn_relu_conv3d(filters=filters, kernel_size=(1, 1, 1),
		                               strides=strides,
		                               kernel_regularizer=kernel_regularizer
		                               )(input)

		conv_3_3 = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
		                           kernel_regularizer=kernel_regularizer
		                           )(conv_1_1)
		residual = _bn_relu_conv3d(filters=filters * 4, kernel_size=(1, 1, 1),
		                           kernel_regularizer=kernel_regularizer
		                           )(conv_3_3)

		return _shortcut3d(input, residual)


	def residual_block3d(filters, kernel_regularizer, repetitions, is_first_layer=False):
		for i in range(repetitions):
		    strides = (1, 1, 1)
		    if i == 0 and not is_first_layer:
		        strides = (2, 2, 2)
		    input = bottleneck(filters=filters, strides=strides,kernel_regularizer=kernel_regularizer,is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
		return input



	# This function defines the baseline model in Keras
	def build_model(self, input_shape):
		reg_factor = 1e-4
		repetitions = [3, 4, 6, 3]
		# Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
		X_input = Input(input_shape)
		
		D1 = conv_3d(filters=64, kernel_size=(7, 7, 7), strides=(2, 2, 2), kernel_regularizer=l2(reg_factor))(input)
	
		D2 =  MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(D1)


		D3 = residual_block3d(filters=64,kernel_regularizer=l2(reg_factor),repetitions=repetitions[0], is_first_layer = True)(D2)
	   	D3 = residual_block3d(filters=128,kernel_regularizer=l2(reg_factor),repetitions=repetitions[1], is_first_layer = False)(D3)
	   	D3 = residual_block3d(filters=256,kernel_regularizer=l2(reg_factor),repetitions=repetitions[2], is_first_layer = False)(D3)
	   	D3 = residual_block3d(filters=512,kernel_regularizer=l2(reg_factor),repetitions=repetitions[3], is_first_layer = False)(D3)

	   	#Axis could be 1
	   	D4 = BatchNormalization(axis=4)(D3)
	   	D4 = Activation("relu")(D4)

	   	D5 = AveragePooling3D(pool_size=(D3._keras_shape[1], D3._keras_shape[2], D3._keras_shape[3]),strides=(1,1,1), padding="same")(D4)
	   	D5 = Flatten()(D5)

	   	
	   	#pred = Dense(units=num_outputs, kernel_initializer="he_normal", activation="sigmoid", kernel_regularizer=l2(reg_factor))(D5)
	    
	   	pred = Conv3D(filters=1, kernel_size=1, activation='sigmoid')(D5)
	   	model = Model(inputs=X_input, outputs=pred, name='ResNet50')

		self.model = model

	def compile(self):
		self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
		self.model.summary()

	def fit(self, X_train, Y_train, validation_set=None):
		earlystopper = EarlyStopping(patience=5, verbose=1)
		checkpointer = ModelCheckpoint('model-resnet50-1', verbose=1, save_best_only=True)
		if validation_set is None:
			return self.model.fit(x=X_train, y=Y_train, validation_split=0.2, epochs=self.epochs, batch_size=self.batch_size, callbacks=[earlystopper, checkpointer])
		else:
			return self.model.fit(x=X_train, y=Y_train, validation_data=validation_set, epochs=self.epochs, batch_size=self.batch_size, callbacks=[earlystopper, checkpointer])

	def evaluate(self, X_test, Y_test):
		return self.model.evaluate(x=X_test, y=Y_test)


