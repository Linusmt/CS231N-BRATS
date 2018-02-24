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

def baseline():
	print('=' * 80)
	print('BASELINE MODEL')
	print('=' * 80)

	mri_path = '../BRATS/Training/HGG/**/*T1c*.mha'
	labels_path = '../BRATS/Training/HGG/**/*OT*.mha'
	mri_path = '/tmp/data/baseline_mris.npy'
	labels_path = '/tmp/data/baseline_labels.npy'	

	image_size = (32, 32, 32)
	mri_path = './baseline_mris.npy'
	labels_path = './baseline_labels.npy'
	mris, labels = utils.get_brats_data_shuffled(mri_path, labels_path, image_size, 'baseline', True, False)


	model = BaselineModel(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['binary_accuracy',  utils.mean_iou, utils.brats_f1_score], epochs=200, batch_size=16)
	model.build_model(mris.shape[1:])
	return model, mris, labels

def u3d():
	print('=' * 80)
	print('Unet3DModel MODEL')
	print('=' * 80)

	image_size = (64, 64, 64)
	# mri_path = './BRATS/Training/HGG/**/*T1c*.mha'
	# labels_path = './BRATS/Training/HGG/**/*OT*.mha'
	image_size = (32, 32, 32)
	mri_path = './baseline_mris.npy'
	labels_path = './baseline_labels.npy'

	mris, labels = utils.get_brats_data_shuffled(mri_path, labels_path, image_size, 'baseline', True, False)

	model = Unet3DModel(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['binary_accuracy',  utils.mean_iou, utils.brats_f1_score], epochs=200, batch_size=4)
	model.build_model(mris.shape[1:])
	return model, mris, labels


def u3d_inception():
	print('=' * 80)
	print('Unet3D_Inception MODEL')
	print('=' * 80)

	image_size = (64, 64, 64)
	# mri_path = './BRATS/Training/HGG/**/*T1c*.mha'
	# labels_path = './BRATS/Training/HGG/**/*OT*.mha'
	mri_path = '/tmp/data/baseline_mris.npy'
	labels_path = '/tmp/data/baseline_labels.npy'
	mris, labels = utils.get_brats_data_shuffled(mri_path, labels_path, image_size, 'baseline', True, False)

	model = Unet3DModelInception(optimizer=Adam(1e-4), loss=utils.brats_f1_loss, metrics=['accuracy'], epochs=200, batch_size=16)
	model.build_model(mris.shape[1:])
	model.summary()
	return model, mris, labels
# Main function will run and test different models
def main():
	model, mris, labels = baseline()

	model.compile()
	history = model.fit(mris, labels)
	utils.plot(history, model_name)


if __name__ == '__main__':
	main()