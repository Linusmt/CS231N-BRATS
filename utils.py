import keras.backend as kb
import skimage.io as io
import skimage.transform as transform
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg') # to save plots as images
import matplotlib.pyplot as plt
import tensorflow as tf
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

def get_brats_data(mri_path, labels_path, image_size, model_name='unknown', preprocessed=False, save=False, shuffle=True):
	# Load data
	mris = brats_load_data(mri_path, 'mri', preprocessed)
	labels = brats_load_data(labels_path, 'labels', preprocessed)
	# Preprocess data (and save for later use)
	if not preprocessed:
		mris = brats_preprocess_mri(mris, image_size, model_name + '_mris', save)
		labels = brats_preprocess_labels(labels, image_size, model_name + '_labels', save)


	if shuffle:
		order = np.random.permutation(mris.shape[0])
		mris, labels =  mris[order,], labels[order,]
	# Return data and labels
	return mris, labels


def brats_f1_score(true, prediction):
	prediction = kb.round(prediction)
	intersection = kb.sum(kb.flatten(true) * kb.flatten(prediction))
	union = kb.sum(kb.flatten(true)) + kb.sum(kb.flatten(prediction))
	return (2 * intersection + 1) / (union + 1)

def f1(batch_size):
	def brats_f1_score(true, prediction):
		prediction = kb.round(prediction)
		intersection = kb.sum(kb.flatten(true) * kb.flatten(prediction))
		union = kb.sum(kb.flatten(true)) + kb.sum(kb.flatten(prediction))
		return (2 * intersection + 1) / (union + 1) * batch_size
	return brats_f1_score


def weighted_cross_entropy_loss(pos_weight):

	def _weighted_cross_entropy_loss(y_true, y_pred):
		return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight)
		# pos_loss = kb.flatten(y_true) * (-tf.log(tf.sigmoid(kb.flatten(y_pred))))*weights[1] 
		# neg_loss = (1-kb.flatten(y_true))* (-tf.log(1- tf.sigmoid(kb.flatten(y_pred)))) * weights[0] 
		# return  kb.mean(pos_loss + neg_loss)		
	return _weighted_cross_entropy_loss


# We want to maximize f1 score (thus minimizing the negative f1 score)
def brats_f1_loss(true, prediction):
    return -brats_f1_score(true, prediction)

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
