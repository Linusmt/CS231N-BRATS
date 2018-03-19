import sys
from keras.optimizers import Adam

import utils
from baseline import BaselineModel
from Unet3DModel import Unet3DModel
from Unet3D_Inception import Unet3DModelInception
from ResNet50 import ResNet50Model

#####
import keras.backend as kb
import tensorflow as tf
import skimage.io as io
import skimage.transform as transform
import glob
import numpy as np
import argparse
import pickle
#####

#MRI_PATH = './BRATS2015_Training/HGG/**/*T1c*.mha'
#LABELS_PATH = './BRATS2015_Training/HGG/**/*OT*.mha'
IMAGE_SIZE = (64, 64, 64)
MRI_PATH = 'baseline_mris.npy'
LABELS_PATH = 'baseline_labels.npy'	

# IMAGE_SIZE = (32, 32, 32)
# MRI_PATH = './baseline_mris_128.npy'
# LABELS_PATH = './baseline_labels_128.npy'
METRICS = ['binary_accuracy',  utils.mean_iou,  utils.brats_f1_score, utils.precision, utils.recall]


MRI_LOAD_PATH = './BRATS/Training/HGG/**/*Flair*.mha'
LABELS_LOAD_PATH = './BRATS/Training/HGG/**/*OT*.mha'
MRI_PATH = 'baseline_mris'
LABELS_PATH = 'baseline_labels'	

METRICS = ['binary_accuracy',  utils.mean_iou,  utils.brats_f1_score, utils.precision, utils.recall]
MODELS = {"baseline":BaselineModel, "u3d":Unet3DModel, "u3d_inception": Unet3DModelInception, "resNet50": ResNet50Model }

def load_data(image_size, preprocess, augment_data=False):

	if preprocess:
		print ("Preprocessing Data Set")
		utils.preproc_brats_data(MRI_LOAD_PATH, LABELS_LOAD_PATH, image_size, 'baseline', save=True)

	#Paths for the MRI data for the given image size 
	mri_path = MRI_PATH + "_" + str(image_size) +".npy"
	labels_path = LABELS_PATH + "_" + str(image_size) +".npy"

	mris, labels = utils.get_brats_data(mri_path, labels_path, image_size, 'baseline', save=False, preprocessed=True, shuffle=False)
	validation_set = (mris[:20,], labels[:20,])
	if augment_data:
		mris, labels  = utils.augment_data(mris[20:,], labels[20:,])
	else:
		mris, labels  = mris[20:,], labels[20:,]
	return mris, labels, validation_set



def main(args):
	#Set the seed for consistent runs
	np.random.seed(42)

	#Take the arguments from the command line
	model_name = args.model
	num_epochs = args.epochs
	print( "Preprocessed: ", args.preprocess)
	mris, labels, validation_set = load_data(args.image_size, args.preprocess, args.augment_data)

	#Create the model
	print (mris.shape)
	print (labels.shape)
	model_generator = MODELS[model_name]
	global_step = tf.Variable(0, name="global_step", trainable=False)
	decay_step = mris.shape[0]/4
	lr = tf.train.exponential_decay(args.lr, global_step, decay_step, 0.96)

	model = model_generator(optimizer=Adam(lr),loss='binary_crossentropy', metrics=METRICS, epochs=num_epochs, batch_size=1, model_name=model_name)
	model.build_model(mris.shape[1:])
	model.compile()

	#Fit the models
	history = model.fit(mris, labels, validation_set=validation_set)

	#Save the model training history for later inspection
	with open("_".join(['./train_history', model_name, str(num_epochs), str(args.image_size),"aug" if args.augment_data else "notaug"])+".pkl", "wb") as history_file:
		pickle.dump(history.history, history_file)

	#Plot the accuracy and the f1 score
	utils.plot(history, model_name, num_epochs, args.image_size)





if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--model', type=str, nargs='?', default="baseline",
	                    help='name of model, possibilities are: baseline, u3d, u3d_inception')
	parser.add_argument('--epochs', type=int, nargs='?', default=100,
	                    help='number of desired epochs')
	parser.add_argument('--preprocess', type=bool,  default=False,
	                    help='whether to load the dataset again and preprocess')	
	parser.add_argument('--image_size', type=int, nargs='?', default=32,
	                    help='new image size to be chosen')
	parser.add_argument('--augment_data', type=bool, nargs='?', default=False,
	                    help='whether to use data augmentation')	
	parser.add_argument('--lr', type=float, nargs='?', default=1e-4,
	                    help='learning rate as a float')
	args = parser.parse_args()

	main(args)