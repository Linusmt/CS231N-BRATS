import sys
from keras.optimizers import Adam

import data_prep.utils as utils
from models import models as models

from keras.callbacks import EarlyStopping, ModelCheckpoint

from pathlib import Path

#####
import keras.backend as kb
from keras.callbacks import Callback as cb
import tensorflow as tf
import glob
import numpy as np
import argparse
import pickle
import data_prep.data_utils as data_utils
#####
import time

IMAGE_SIZE = (64, 64, 64)


# IMAGE_SIZE = (32, 32, 32)
# MRI_PATH = './baseline_mris_128.npy'
# LABELS_PATH = './baseline_labels_128.npy'
METRICS = ['binary_accuracy',  utils.mean_iou,  utils.brats_f1_score, utils.precision, utils.recall]

METRICS = ['binary_accuracy',  utils.mean_iou,  utils.brats_f1_score, utils.precision, utils.recall]

class TimeHistory(cb):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# def continue_training_model(model_name):


def main(args):
	#Set the seed for consistent runs
	np.random.seed(42)

	#Take the arguments from the command line
	model_name = args.model
	num_epochs = args.epochs
	use_dropout = args.use_dropout
	print( "Preprocessed: ", args.preprocess)
	X, y, validation_set = data_utils.load_data(args.image_size, args.preprocess, args.augment_data)

	#Create the model
	if args.test_model:
		X = X[0:5,]
		y = y[0:5,]
		validation_set = (validation_set[0][0:3,], validation_set[1][0:3,])

	print (X.shape)
	print (y.shape)

	model_generator = models.MODELS[model_name]
	global_step = tf.Variable(0, name="global_step", trainable=False)
	decay_step = X.shape[0]/4
	lr = tf.train.exponential_decay(args.lr, global_step, decay_step, 0.98)

	model = model_generator(optimizer=Adam(args.lr),loss='binary_crossentropy', metrics=METRICS, epochs=num_epochs, batch_size=1, model_name=model_name, use_dropout=use_dropout)
	model.build_model(X.shape[1:])
	model.compile()

	history_file_name = "_".join(['model', model_name, str(num_epochs), str(args.image_size), "dropout_" + str(use_dropout) if use_dropout != 0 else "","test" if args.test_model else ""])

	if args.test_model:
		model_save_path = './tmp/' +history_file_name + '.h5'
	else:
		model_save_path = './models/' +history_file_name + '.h5'

	my_file = Path(model_save_path)
	if my_file.is_file() and args.load_weights:
		print("Loading model from path")
		model.model.load_weights(model_save_path)

	checkpointer = ModelCheckpoint(model_save_path, verbose=1, save_best_only=True)


	time_callback = TimeHistory()
	earlystopper = EarlyStopping(patience=20, verbose=1)

	history = model.model.fit(x=X, y=y, validation_data=validation_set, epochs=num_epochs, batch_size=1, callbacks=[checkpointer, time_callback, earlystopper])


	history.history["times"] = time_callback.times
	history.history["flags"] = args


	#Save the model training history for later inspection
	if args.test_model:
		with open("./tmp/" +history_file_name+".pkl", "wb") as history_file:
			pickle.dump(history.history, history_file)
	else:
		with open("./history/" +history_file_name+".pkl", "wb") as history_file:
			pickle.dump(history.history, history_file)

	#Plot the accuracy and the f1 score
	utils.plot(history, model_name, num_epochs, args.image_size, test=True)





if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--model', type=str, nargs='?', default="baseline",
	                    help='name of model, possibilities are: baseline, u3d, u3d_inception')
	parser.add_argument('--epochs', type=int, nargs='?', default=5,
	                    help='number of desired epochs')
	parser.add_argument('--preprocess', type=bool,  default=False,
	                    help='whether to load the dataset again and preprocess')	
	parser.add_argument('--image_size', type=int, nargs='?', default=64,
	                    help='new image size to be chosen')
	parser.add_argument('--augment_data', type=bool, nargs='?', default=False,
	                    help='whether to use data augmentation')	
	parser.add_argument('--lr', type=float, nargs='?', default=1e-4,
	                    help='learning rate as a float')
	parser.add_argument('--test_model', type=bool, nargs="?", default=False,
						help="use a small dataset to make sure everything works ")
	parser.add_argument('--use_dropout', type=float, nargs="?", default=0.0,
						help="amount of dropout to use")
	parser.add_argument('--load_weights', type=bool, nargs="?", default=True,
						help="whether to load in the weights")
	args = parser.parse_args()

	main(args)