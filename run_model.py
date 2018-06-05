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
from data_prep.data_generator import DataGenerator

IMAGE_SIZE = (64, 64, 64)


# IMAGE_SIZE = (32, 32, 32)
# MRI_PATH = './baseline_mris_128.npy'
# LABELS_PATH = './baseline_labels_128.npy'
METRICS = ['binary_accuracy',  utils.mean_iou,  utils.brats_f1_score, utils.precision, utils.recall]

METRICS = ['binary_accuracy',  utils.mean_iou,  utils.brats_f1_score, utils.precision, 
			utils.recall, utils.enhancing_f1_score, utils.non_enhancing_f1_score, utils.edema_f1_score, utils.class_3_f1_score, 
			utils.zero_f1_score]

class TimeHistory(cb):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# def continue_training_model(model_name):
from decimal import Decimal
NUM_CHANNELS = 3
def main(args):
	#Set the seed for consistent runs
	np.random.seed(42)

	#Take the arguments from the command line
	model_name = args.model
	use_dropout = args.use_dropout
	print( "Preprocessed: ", args.preprocess)
	# data_utils.load_data(args.image_size, args.preprocess, args.augment_data)
	if args.preprocess:
		data_utils.preproc_brats_data(args.image_size, 'baseline', save=True)

	# Pick the model specified by model_name
	model_generator = models.MODELS[model_name]

	# Use learning rate decay decreasing at 0.88 ^ epoch
	global_step = tf.Variable(0, name="global_step", trainable=False)
	decay_step = 1900
	lr = tf.train.exponential_decay(args.lr, global_step, decay_step, 0.95)

	model = model_generator(optimizer=Adam(args.lr),loss='categorical_crossentropy', metrics=METRICS, epochs=args.epochs, batch_size=1, model_name=model_name, use_dropout=use_dropout)
	model.build_model([args.image_size,args.image_size, args.image_size, NUM_CHANNELS])
	model.compile()

	dropout_str = "dropout_" + str(use_dropout)
	test_str = "test" if args.test_model else ""
	lr_str = 'lr_%.1E' % Decimal(args.lr)
	weight_decay_str = 'wd_%.2f' % (args.weight_decay)
	augment_str = "aug" if str(args.augment_data) else ""

	history_file_name = "_".join(['model', model_name, str(args.epochs), str(args.image_size), dropout_str, test_str, lr_str, weight_decay_str, augment_str])

	if args.test_model:
		model_save_path = './tmp/' +history_file_name + '.h5'
	else:
		model_save_path = './models/' +history_file_name + '.h5'

	my_file = Path(model_save_path)
	# if my_file.is_file() and args.load_weights:
	# 	print("Loading model from path")
	# 	try {
	# 	model.model.load_weights(model_save_path)
	# 	} 
	checkpointer = ModelCheckpoint(model_save_path, verbose=1, save_best_only=True)
	time_callback = TimeHistory()
	earlystopper = EarlyStopping(patience=20, verbose=1)

	crops_paths, labels_paths = glob.glob("./data/crops/*.npy"), glob.glob("./data/labels/*.npy")
	data_generator = DataGenerator(crops_paths[0:-200], labels_paths[0:-200], args.test_model)
	val_generator = DataGenerator(crops_paths[-200:len(crops_paths)], labels_paths[-200:len(crops_paths)], args.test_model)
	
	if args.augment_data:
		history =  model.model.fit_generator(generator=data_generator, epochs=args.epochs,
                    validation_data=val_generator,
                     callbacks=[checkpointer, time_callback, earlystopper])
	else:

		history = model.model.fit(x=X, y=y, validation_data=validation_set, epochs=args.epochs, batch_size=1, callbacks=[checkpointer, time_callback, earlystopper])

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
	utils.plot(history, model_name, args.epochs, args.image_size, test=True)





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
	parser.add_argument('--lr', type=float, nargs='?', default=1e-3,
	                    help='learning rate as a float')
	parser.add_argument('--test_model', type=bool, nargs="?", default=False,
						help="use a small dataset to make sure everything works ")
	parser.add_argument('--use_dropout', type=float, nargs="?", default=0.0,
						help="amount of dropout to use")
	parser.add_argument('--load_weights', type=bool, nargs="?", default=True,
						help="whether to load in the weights")
	parser.add_argument('--weight_decay', type=float, nargs="?", default=1.0,
						help="whether to load in the weights")
	args = parser.parse_args()

	main(args)