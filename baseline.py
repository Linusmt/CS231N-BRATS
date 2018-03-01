import numpy as np
import pandas as pd
import random
import keras
from keras.optimizers import Adam
from keras.models import Input, Model
from keras.layers import Conv3D, Concatenate, MaxPooling3D, AveragePooling3D, UpSampling3D, Activation, Reshape, Permute
from keras.callbacks import EarlyStopping, ModelCheckpoint


class BaselineModel():
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


	# This function defines the baseline model in Keras
	def build_model(self, input_shape):

		# Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
	    X_input = Input(input_shape)

	    # [1] First CONV--POOL--ReLU layer
	    X = Conv3D(filters=16, kernel_size=3, strides=(1,1,1), padding='same', activation='elu')(X_input)
	    X = AveragePooling3D(strides=(1,1,1), padding="same")(X)

	    # [2] Second CONV--POOL--ReLU layer
	    X = Conv3D(filters=32, kernel_size=3, strides=(1,1,1), padding='same', activation='elu')(X)
	    X = AveragePooling3D(strides=(1,1,1), padding="same")(X)

	    # [3] Third CONV--POOL--ReLU layer
	    X = Conv3D(filters=64, kernel_size=3, strides=(1,1,1), padding='same', activation='elu')(X)
	    X = AveragePooling3D(strides=(1,1,1), padding="same")(X)

	    # Final prediction (sigmoid)
	    X = Conv3D(filters=1, kernel_size=1, activation='sigmoid')(X)

	    # Create instance of Baseline Model
	    model = Model(inputs=X_input, outputs=X, name='BaselineModel')

	    self.model = model

	def compile(self):
		self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
		self.model.summary()

	def fit(self, X_train, Y_train, validation_set=None):
		earlystopper = EarlyStopping(patience=5, verbose=1)
		checkpointer = ModelCheckpoint('model-baseline-1', verbose=1, save_best_only=True)
		if validation_set is None:
			return self.model.fit(x=X_train, y=Y_train, validation_split=0.2, epochs=self.epochs, batch_size=self.batch_size, callbacks=[earlystopper, checkpointer])
		else:
			return self.model.fit(x=X_train, y=Y_train, validation_data=validation_set, epochs=self.epochs, batch_size=self.batch_size, callbacks=[earlystopper, checkpointer])


	def evaluate(self, X_test, Y_test):
		return self.model.evaluate(x=X_test, y=Y_test)






