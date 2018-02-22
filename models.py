import sys
from keras.optimizers import Adam

import utils
from baseline import BaselineModel

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

	image_size = (32, 32, 32)
	# mri_path = './BRATS/Training/HGG/**/*T1c*.mha'
	# labels_path = './BRATS/Training/HGG/**/*OT*.mha'
	mri_path = 'baseline_mris.npy'
	labels_path = 'baseline_labels.npy'
	mris, labels = utils.get_brats_data(mri_path, labels_path, image_size, 'baseline', True, False)

	model = BaselineModel(optimizer=Adam(1e-5), loss=utils.brats_f1_loss, metrics=['accuracy'], epochs=10, batch_size=16)
	model.build_model(mris.shape[1:])
	return model, mris, labels


# Main function will run and test different models
def main():
	model, mris, labels = baseline()

	model.compile()
	history = model.fit(mris, labels)
	utils.plot(history, model_name)


if __name__ == '__main__':
	main()