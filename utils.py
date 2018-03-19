import keras.backend as kb
import skimage.io as io
import skimage.transform as transform
import glob
import numpy as np
# import matplotlib
# matplotlib.use('Agg') # to save plots as images
import matplotlib.pyplot as plt
import tensorflow as tf
import math

MRI_FLAIR_LOAD_PATH = './BRATS/Training/HGG/**/*Flair*.mha'
MRI_T1c_PATH = './BRATS/Training/HGG/**/*T1c*.mha'
MRI_T2_LOAD_PATH = './BRATS/Training/HGG/**/*T2*.mha'
DATA_PATHS = [MRI_T1c_PATH, MRI_T2_LOAD_PATH, MRI_FLAIR_LOAD_PATH]
LABELS_LOAD_PATH = './BRATS/Training/HGG/**/*OT*.mha'

############################# BRATS #################################
def resize_by_slice_mri(image, newsize):
	newsize = [newsize]*2
	depth = math.floor(image.shape[0] / 8) * 8
	image = np.array([ transform.resize(image[i, :, :], newsize, mode='constant') for i in range(image.shape[0]) ])
	return transform.resize(image, [depth] + newsize , mode='constant')

def resize_by_slice_label(image, newsize):
	newsize = [newsize]*2
	depth = math.floor(image.shape[0] / 8) * 8

	np.array([ transform.resize(image[i,:,:] , newsize, mode='constant', preserve_range=True) for i in range(image.shape[0]) ])
	return transform.resize(image, [depth] + newsize , mode='constant', preserve_range=True)


def brats_load_data(path, dataname='unknown', preprocessed=False):
	print('Loading ' + dataname + ' data...')
	# print( glob.glob(path, recursive=True))
	if not preprocessed:
		return [ io.imread(f, plugin='simpleitk') for f in glob.glob(path, recursive=True) ]
	else:
		return np.load(path)

def brats_preprocess_mri(images, newsize, name='unknown', save=False):
	preprocessed = [ resize_by_slice_mri((i - i.mean()) / i.std(), newsize) for i in images ]
	return np.array(preprocessed)[..., np.newaxis].astype('float32')

def brats_preprocess_labels(images, newsize, name='unknown', save=False):
	preprocessed = []
	for i in images:
		i[i != 4] = 0
		i[i == 4] = 1
		preprocessed.append(resize_by_slice_label(i, newsize))
	if save:
		if name == 'unknown': print('Cannot save file unless new filename is specified')
		else: np.save(name, np.array(preprocessed)[..., np.newaxis].astype('float32'))
	return np.array(preprocessed)[..., np.newaxis].astype('float32')

def preproc_brats_data(mri_path, labels_path, image_size, model_name='unknown',  save=False):
	channels = []
	for path in DATA_PATHS:
		mris = brats_load_data(path, 'mri', preprocessed)
		mris = brats_preprocess_mri(mris, image_size, model_name + '_mris_' +str(image_size), save)
		# mris = np.expand_dims(mris, axis=-1)
		channels.append(mris)

	mris = np.squeeze(np.stack(channels, axis=4), axis=5)
	print(mris.shape)
	np.save(model_name + '_mris_' +str(image_size), np.array(mris).astype('float32'))


	labels = brats_load_data(labels_path, 'labels', preprocessed)
	labels = brats_preprocess_labels(labels, image_size, model_name + '_labels_' +str(image_size), save)


def get_brats_data(mri_path, labels_path, image_size, model_name='unknown', preprocessed=False, save=False, shuffle=True):
	# Load data
	mris = np.load(mri_path)
	labels =  np.load(labels_path)
	# Preprocess data (and save for later use)


	# Return data and labels
	return mris, labels

def augment_data(X, Y):
	new_X = [X, np.flip(X, axis=1), np.flip(X, axis=2), np.flip(X, axis=3)]
	new_Y = [Y, np.flip(Y, axis=1), np.flip(Y, axis=2), np.flip(Y, axis=3)]
	new_X = np.concatenate(new_X, axis=0)
	new_Y = np.concatenate(new_Y, axis=0)
	return new_X,new_Y

def brats_f1_score(true, prediction):
	prediction = kb.round(prediction)
	intersection = kb.sum(kb.flatten(true) * kb.flatten(prediction))
	union = kb.sum(kb.flatten(true)) + kb.sum(kb.flatten(prediction))
	return kb.clip((2 * intersection + 1) / (union + 1), 1.0e-8, 1)

def brats_f1_score_loss(true, prediction):
	intersection = kb.sum(kb.flatten(true) * kb.flatten(prediction))
	union = kb.sum(kb.flatten(true)) + kb.sum(kb.flatten(prediction))
	return kb.clip((2 * intersection + 1) / (union + 1), 1.0e-8, 1)

def precision(true, prediction):
	prediction = kb.round(prediction)
	intersection = kb.sum(kb.flatten(true) * kb.flatten(prediction))
	union = kb.sum(kb.flatten(prediction))
	return kb.clip((2 * intersection + 1) / (union + 1), 1.0e-8, 1)

def recall(true, prediction):
	prediction = kb.round(prediction)
	intersection = kb.sum(kb.flatten(true) * kb.flatten(prediction))
	union = kb.sum(kb.flatten(true))
	return kb.clip((2 * intersection + 1) / (union + 1), 1.0e-8, 1)

# def brats_f1_score(true, prediction):
# 	prediction = kb.round(prediction)
# 	intersection = kb.sum(kb.flatten(true) * kb.flatten(prediction))
# 	union = kb.sum(kb.flatten(true)) + kb.sum(kb.flatten(prediction))
# 	return tf.get_variable("f1_score", initializer=(2 * intersection + 1) / (union + 1) )
# return brats_f1_score


def weighted_cross_entropy_loss(pos_weight):

	def _weighted_cross_entropy_loss(y_true, y_pred):
		return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight)
		# pos_loss = kb.flatten(y_true) * (-tf.log(tf.sigmoid(kb.flatten(y_pred))))*weights[1] 
		# neg_loss = (1-kb.flatten(y_true))* (-tf.log(1- tf.sigmoid(kb.flatten(y_pred)))) * weights 
		# return  kb.mean(pos_loss + neg_loss)		
	return _weighted_cross_entropy_loss


# We want to maximize f1 score (thus minimizing the negative f1 score)
def brats_f1_loss(true, prediction):
    return -brats_f1_score_loss(true, prediction)

####################################################################

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2, y_true)
        kb.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return kb.mean(kb.stack(prec), axis=0)


def plot(history, model_name, num_epochs, image_size):
	# Create plot that plots model accuracy vs. epoch
	print("Plotting accuracy")
	fig = plt.figure(figsize=(10, 10))
	plt.plot(history.history['binary_accuracy'])
	plt.plot(history.history['val_binary_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('./output/accuracy-' + model_name +"-" + str(num_epochs) +"-" + str(image_size) + '-history1.png'.format(32))
	print("Finished plotting accuracy")
	plt.close(fig)
	print("Plotting f1_score")

	fig = plt.figure(figsize=(10, 10))

	plt.plot(history.history['brats_f1_score'])
	plt.plot(history.history['val_brats_f1_score'])
	plt.title('f1 score')
	plt.ylabel('f1 score')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('./output/f1-' + model_name +"-" + str(num_epochs) + "-" + str(image_size) + '-history1.png'.format(32))
	plt.close(fig)

	print("Finished plotting f1_score")

	fig = plt.figure(figsize=(10, 10))

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('./output/loss-' + model_name +"-" + str(num_epochs) + "-" + str(image_size) + '-history1.png'.format(32))
	plt.close(fig)

	print("Finished plotting loss")