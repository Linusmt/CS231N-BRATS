import numpy as np
import pandas as pd
import random
import keras
from keras.optimizers import Adam
from keras.models import Input, Model
from keras.layers import Conv3D,BatchNormalization, Concatenate, MaxPooling3D, AveragePooling3D, UpSampling3D, Activation, Reshape, Permute, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

class Unet3DModelInception():
	# Params:
	# optimizer: keras optimizer (e.g. Adam)
	# loss: loss function for optimization
	# metrics: list of metrics (e.g. ['accuracy'])
	def __init__(self, optimizer, loss, metrics=['accuracy'], epochs=1, batch_size=16, model_name="unknown", use_dropout=0.0):
		self.optimizer = optimizer
		self.model_name = model_name
		self.loss = loss
		self.metrics = metrics
		self.epochs = epochs
		self.batch_size = batch_size
		self.model = None
		self.dropout = use_dropout



	def inception_layer(self, X, filters):
		X1 = Conv3D(filters=filters[0], kernel_size=1, strides=(1,1,1), padding='same', activation='tanh')(X)

		X33 = Conv3D(filters=filters[1][0], kernel_size=1, strides=(1,1,1), padding='same', activation='tanh')(X)
		X33 = Conv3D(filters=filters[1][1], kernel_size=3, strides=(1,1,1), padding='same', activation='tanh')(X33)

		X55 = Conv3D(filters=filters[2][0], kernel_size=1, strides=(1,1,1), padding='same', activation='tanh')(X)
		X55 = Conv3D(filters=filters[2][1], kernel_size=5, strides=(1,1,1), padding='same', activation='tanh')(X55)

		X_Max = MaxPooling3D(pool_size=(3,3,3),strides=(1,1,1), padding="same")(X)
		X_Max = Conv3D(filters=filters[3], kernel_size=1, strides=(1,1,1), padding='same', activation='tanh')(X_Max)

		out_layer = Concatenate()([X1, X33, X55, X_Max])
		out_layer = Dropout(self.dropout)(out_layer)
		return out_layer

	# This function defines the baseline model in Keras
	def build_model(self, input_shape):

		# Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
	    X_input = Input(input_shape)
	    # X = Conv3D(filters=16, kernel_size=3, strides=(1,1,1), padding='same', activation='tanh')(X_input)
	    X = Conv3D(filters=32, kernel_size=3, strides=(1,1,1), padding='same', activation='tanh')(X_input)
	    D1 = self.inception_layer(X, filters=[16, [16, 32], [4, 8], 16])

	    D2 = AveragePooling3D(strides=(2,2,2), padding="same")(D1)
	    D2 = self.inception_layer(D2, filters=[16, [16, 32], [8, 16], 16])

	    D3 = AveragePooling3D(strides=(2,2,2), padding="same")(D2)
	    # D3 = self.inception_layer(D3, filters=[64, [64, 128], [16, 32], 64])
	    D3 = self.inception_layer(D3, filters=[32, [32, 64], [8, 32], 32])

	    D4 = AveragePooling3D(strides=(2,2,2), padding="same")(D3)
	    D4 = self.inception_layer(D4, filters=[96, [64, 128], [32, 64], 64])

	    U3 = Conv3D(256, 2, padding='same', activation='tanh')(UpSampling3D(size = (2,2,2), dim_ordering="tf")(D4))
	    U3 = Concatenate()( [D3, U3])
	    U3 = self.inception_layer(U3, filters=[32, [32, 64], [16, 32], 32])


	    U2 = Conv3D(128, 2, padding='same', activation='tanh')(UpSampling3D(size = (2,2,2), dim_ordering="tf")(U3))
	    U2 = Concatenate()( [D2, U2])
	    U2 = self.inception_layer(U2, filters=[32, [32, 64], [16, 32], 32])

	    U1 = Conv3D(64, 2, padding='same', activation='tanh')(UpSampling3D(size = (2,2,2), dim_ordering="tf")(U2))
	    U1 = Concatenate()( [D1, U1])
	    U1 = self.inception_layer(U1, filters=[16, [16, 32], [16, 32], 16])

	    pred = Conv3D(filters=5, kernel_size=1, activation='softmax')(U1)


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


