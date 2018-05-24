import sys
from keras.optimizers import Adam

import data_prep.utils as utils
from baseline import BaselineModel
from Unet3DModel import Unet3DModel
from Unet3D_Inception import Unet3DModelInception
from ResNet50 import ResNet50Model

#####
import keras.backend as kb
import tensorflow as tf
import glob
import numpy as np
import argparse
import pickle
import data_prep.data_utils as data_utils
#####

IMAGE_SIZE = (64, 64, 64)


# IMAGE_SIZE = (32, 32, 32)
# MRI_PATH = './baseline_mris_128.npy'
# LABELS_PATH = './baseline_labels_128.npy'
METRICS = ['binary_accuracy',  utils.mean_iou,  utils.brats_f1_score, utils.precision, utils.recall]

METRICS = ['binary_accuracy',  utils.mean_iou,  utils.brats_f1_score, utils.precision, utils.recall]
MODELS = {"baseline":BaselineModel, "u3d":Unet3DModel, "u3d_inception": Unet3DModelInception, "resNet50": ResNet50Model }



def main(args):
	#Set the seed for consistent runs
	np.random.seed(42)

	#Take the arguments from the command line
	model_name = args.model
	num_epochs = args.epochs
	print( "Preprocessed: ", args.preprocess)
	X, y, validation_set = data_utils.load_data(args.image_size, args.preprocess, args.augment_data)

	#Create the model
	if args.test_model:
		X = X[0:10,]
		y = y[0:10,]

	print (X.shape)
	print (y.shape)

	model_generator = MODELS[model_name]
	global_step = tf.Variable(0, name="global_step", trainable=False)
	decay_step = X.shape[0]/4
	lr = tf.train.exponential_decay(args.lr, global_step, decay_step, 0.96)

	model = model_generator(optimizer=Adam(lr),loss='binary_crossentropy', metrics=METRICS, epochs=num_epochs, batch_size=1, model_name=model_name)
	model.build_model(X.shape[1:])
	model.compile()

	#Fit the models
	history = model.fit(X, y, validation_set=validation_set)

	#Save the model training history for later inspection
	with open("_".join(['./train_history', model_name, str(num_epochs), str(args.image_size),"test" if args.test_model else ""])+".pkl", "wb") as history_file:
		pickle.dump(history.history, history_file)

	#Plot the accuracy and the f1 score
	utils.plot(history, model_name, num_epochs, args.image_size)





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
	args = parser.parse_args()

	main(args)