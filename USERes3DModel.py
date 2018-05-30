import numpy as np
import pandas as pd
import random
import keras
from keras.optimizers import Adam
from keras.models import Input, Model
from keras.layers import Conv3D,BatchNormalization, Concatenate, MaxPooling3D, AveragePooling3D, UpSampling3D, Activation, Reshape, Permute, Add, Multiply
from keras.callbacks import EarlyStopping, ModelCheckpoint
from SE_module import Squeeze_excitation_layer

class USERes3DModel():
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
		self.Squeeze_excitation_layer = Squeeze_excitation_layer



	def double_block(self, X, f, kernel_size, s):
		X1 = Conv3D(filters=f[0], kernel_size=kernel_size, strides=(s,s,s), padding='same', activation='elu')(X)

		X1 = self.Squeeze_excitation_layer(X1, f[1], 16)

		X1 = BatchNormalization(axis = 4)(X1)

		X1 = Conv3D(filters=f[1], kernel_size=kernel_size, strides=(s,s,s), padding='same', activation='elu')(X1)

		X1 = self.Squeeze_excitation_layer(X1, f[1], 16)

		X1 = BatchNormalization(axis = 4)(X1)

		return X1

	# This function defines the baseline model in Keras
	def build_model(self, input_shape):

		# Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
	    X_input = Input(input_shape)

	    D1 = self.double_block(X_input, [64,64], 3, 1)

	    D2_in = AveragePooling3D(strides=(2,2,2), padding="same")(D1)
	    D2_out = self.double_block(D2_in, [128, 128], 3, 1)
	    D2_in = Conv3D(128, kernel_size=1, strides=(1,1,1), padding='same')(D2_in)
	    D2 = Add()([D2_in, D2_out])

	    D3_in = AveragePooling3D(strides=(2,2,2), padding="same")(D2)
	    D3_out = self.double_block(D3_in, [256,256], 3, 1)
	    D3_in = Conv3D(256, kernel_size=1, strides=(1,1,1), padding='same')(D3_in)
	    D3 = Add()([D3_in, D3_out])

	    D4_in = AveragePooling3D(strides=(2,2,2), padding="same")(D3)
	    D4_out = self.double_block(D4_in, [512,512], 3, 1)
	    D4_in = Conv3D(512, kernel_size=1, strides=(1,1,1), padding='same')(D4_in)
	    D4 = Add()([D4_in, D4_out])

	    U3_in = Conv3D(512, 2, padding='same')(UpSampling3D(size = (2,2,2), dim_ordering="tf")(D4))
	    U3 = Activation('elu')(U3_in)
	    U3 = Concatenate()( [D3, U3])
	    U3_out = self.double_block(U3, [256, 256], 3 ,1 )
	    U3_in = Conv3D(256, kernel_size=1, strides=(1,1,1), padding='same')(U3_in)
	    U3 = Add()([U3_in, U3_out])


	    U2_in = Conv3D(256, 2, padding='same')(UpSampling3D(size = (2,2,2), dim_ordering="tf")(U3))
	    U2 = Activation('elu')(U2_in)
	    U2 = Concatenate()( [D2, U2])
	    U2_out = self.double_block(U2, [128, 128], 3 ,1 )
	    U2_in = Conv3D(128, kernel_size=1, strides=(1,1,1), padding='same')(U2_in)

	    U2 = Add()([U2_in, U2_out])

	    U1_in = Conv3D(128, 2, padding='same')(UpSampling3D(size = (2,2,2), dim_ordering="tf")(U2))
	    U1 = Activation('elu')(U1_in)
	    U1 = Concatenate()( [D1, U1])
	    U1_out = self.double_block(U1, [64, 64], 3 ,1 )
	    U1_in = Conv3D(64, kernel_size=1, strides=(1,1,1), padding='same')(U1_in)

	    U1 = Add()([U1_in, U1_out])

	    pred = Conv3D(filters=1, kernel_size=1, activation='sigmoid')(U1)

	    # Create instance of Baseline Model
	    model = Model(inputs=X_input, outputs=pred, name='BaselineModel')

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
				return self.model.fit(x=X_train, y=Y_train, validation_data=validation_set, epochs=self.epochs, batch_size=self.batch_size, callbacks=[ checkpointer], global_step=global_step)
			else:
				return self.model.fit(x=X_train, y=Y_train, validation_data=validation_set, epochs=self.epochs, batch_size=self.batch_size, callbacks=[ checkpointer])

	def evaluate(self, X_test, Y_test):
		return self.model.evaluate(x=X_test, y=Y_test)


