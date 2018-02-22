import keras.backend as kb
import skimage.io as io
import skimage.transform as transform
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg') # to save plots as images
import matplotlib.pyplot as plt

############################# BRATS #################################

def brats_load_data(path, dataname='unknown', preprocessed=False):
	print('Loading ' + dataname + ' data...')
	if not preprocessed:
		return [ io.imread(f, plugin='simpleitk') for f in glob.glob(path, recursive=True) ]
	else:
		return np.load(path)

def brats_preprocess_mri(images, newsize, name='unknown', save=False):
	preprocessed = [ transform.resize((i - i.mean()) / i.std(), newsize, mode='constant') for i in images ]
	if save:
		if name == 'unknown': print('Cannot save file unless new filename is specified')
		else: np.save(name, np.array(preprocessed)[..., np.newaxis].astype('float32'))
	return np.array(preprocessed)[..., np.newaxis].astype('float32')

def brats_preprocess_labels(images, newsize, name='unknown', save=False):
	preprocessed = []
	for i in images:
		i[i == 4] = 1
		i[i != 1] = 0
		preprocessed.append(transform.resize(i, newsize, mode='constant').astype('float32'))
	if save:
		if name == 'unknown': print('Cannot save file unless new filename is specified')
		else: np.save(name, np.array(preprocessed)[..., np.newaxis].astype('float32'))
	return np.array(preprocessed)[..., np.newaxis].astype('float32')

def get_brats_data(mri_path, labels_path, image_size, model_name='unknown', preprocessed=False, save=False):
	# Load data
	mris = brats_load_data(mri_path, 'mri', preprocessed)
	labels = brats_load_data(labels_path, 'labels', preprocessed)
	# Preprocess data (and save for later use)
	if not preprocessed:
		mris = brats_preprocess_mri(mris, image_size, model_name + '_mris', save)
		labels = brats_preprocess_labels(labels, image_size, model_name + '_labels', save)

	# Return data and labels
	return mris, labels

def brats_f1_score(true, prediction):
    intersection = kb.sum(kb.flatten(true) * kb.flatten(prediction))
    union = kb.sum(kb.flatten(true)) + kb.sum(kb.flatten(prediction))
    return (2 * intersection + 1) / (union + 1)

# We want to maximize f1 score (thus minimizing the negative f1 score)
def brats_f1_loss(true, prediction):
    return -brats_f1_score(true, prediction)

####################################################################


def plot(history, model_name):
	# Create plot that plots model accuracy vs. epoch
	fig = plt.figure(figsize=(10, 10))
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('/output/size{0}-' + model_name + '-history1.png'.format(32))
	plt.close(fig)
