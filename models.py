import sys
from keras.optimizers import Adam

import utils
from baseline import BaselineModel
from Unet3DModel import Unet3DModel
from Unet3D_Inception import Unet3DModelInception

#####
import keras.backend as kb
import skimage.io as io
import skimage.transform as transform
import glob
import numpy as np
#####

# MRI_PATH = './BRATS2015_Training/HGG/**/*T1c*.mha'
# LABELS_PATH = './BRATS2015_Training/HGG/**/*OT*.mha'
IMAGE_SIZE = (64, 64, 64)
MRI_PATH = 'baseline_mris_64.npy'
LABELS_PATH = 'baseline_labels_64.npy'	

# IMAGE_SIZE = (32, 32, 32)
# MRI_PATH = './baseline_mris_128.npy'
# LABELS_PATH = './baseline_labels_128.npy'
METRICS = ['binary_accuracy',  utils.mean_iou,  utils.brats_f1_score]

def baseline(num_epochs=40):
	print('=' * 80)
	print('BASELINE MODEL')
	print('=' * 80)

	# mris, labels = utils.get_brats_data(MRI_PATH, LABELS_PATH, IMAGE_SIZE, 'baseline', False, True, shuffle=True)


	mris, labels = utils.get_brats_data(MRI_PATH, LABELS_PATH, IMAGE_SIZE, 'baseline', True, False, shuffle=True)

	validation_set = (mris[:20,], labels[:20])
	mris, labels  = (mris[20:,], labels[20:,])

	model = BaselineModel(optimizer=Adam(1e-4),loss='binary_crossentropy', metrics=METRICS, epochs=num_epochs, batch_size=1)
	model.build_model(mris.shape[1:])

	return model, mris, labels, validation_set

def u3d(num_epochs=40):
	print('=' * 80)
	print('Unet3DModel MODEL')
	print('=' * 80)

	mris, labels = utils.get_brats_data(MRI_PATH, LABELS_PATH, IMAGE_SIZE, 'u3d', True, False, shuffle=True)
	
	validation_set = (mris[:20,], labels[:20])
	mris, labels  = utils.augment_data(mris[20:,], labels[20:,])


	model = Unet3DModel(optimizer=Adam(1e-4),loss='binary_crossentropy', metrics=METRICS, epochs=num_epochs, batch_size=1)
	model.build_model(mris.shape[1:])
	return model, mris, labels, validation_set


def u3d_inception(num_epochs=40):
	print('=' * 80)
	print('Unet3D_Inception MODEL')
	print('=' * 80)

	mris, labels = utils.get_brats_data(MRI_PATH, LABELS_PATH, IMAGE_SIZE, 'u3d_inception', True, False, shuffle=True)

	validation_set = (mris[:20,], labels[:20])
	mris, labels  = utils.augment_data(mris[20:,], labels[20:,])

	model = Unet3DModelInception(optimizer=Adam(1e-4),loss='binary_crossentropy', metrics=METRICS, epochs=num_epochs, batch_size=1)
	model.build_model(mris.shape[1:])
	return model, mris, labels, validation_set
# Main function will run and test different models

import argparse


MODELS = {"baseline":baseline, "u3d":u3d, "u3d_inception": u3d_inception }

def main(args):
	#Set the seed for consistent runs
	np.random.seed(42)

	#Take the arguments from the command line
	model_name = args.model
	num_epochs = args.num_epochs

	#Create the model
	model, mris, labels, validation_set = MODELS[model_name](num_epochs=num_epochs)
	model.compile()
	# print (validation_set)
	history = model.fit(mris, labels, validation_set=validation_set)

	#Plot the accuracy and the f1 score
	utils.plot(history, model_name, num_epochs)

	#Save the model history for later inspection
	with open('./train_history_' + model_name + "_" + str(num_epochs), "wb") as history_file:
		pickle.dump(history.history, file_pi)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--model', type=str, nargs='?', default="baseline",
	                    help='name of model')
	parser.add_argument('--num_epochs', type=int, nargs='?', default=10,
	                    help='number of desired epochs')
	args = parser.parse_args()

	main(args)