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
import nibabel as nib

def f1(true, prediction):
	prediction = kb.round(prediction)
	intersection = kb.sum(kb.flatten(true) * kb.flatten(prediction))
	union = kb.sum(kb.flatten(true)) + kb.sum(kb.flatten(prediction))
	return kb.clip((2 * intersection + 1) / (union + 1), 1.0e-8, 1)


def brats_f1_score(true, prediction):
	return f1(true, prediction)

def enhancing_f1_score(true, prediction):
	true = true[:,:,:,:,4]
	prediction = prediction[:,:,:,:,4]
	return f1(true, prediction)

def edema_f1_score(true, prediction):
	true = true[:,:,:,:,2]
	prediction = prediction[:,:,:,:,2]
	return f1(true, prediction)

def non_enhancing_f1_score(true, prediction):
	true = true[:,:,:,:,1]
	prediction = prediction[:,:,:,:,1]
	return f1(true, prediction)

def brats_f1_score_loss(true, prediction):
	intersection = kb.sum(kb.flatten(true) * kb.flatten(prediction))
	union = kb.sum(kb.flatten(true)) + kb.sum(kb.flatten(prediction))
	return kb.clip((2 * intersection + 1) / (union + 1), 1.0e-8, 1)

def precision(true, prediction):
	prediction = kb.round(prediction)
	intersection = kb.sum(kb.flatten(true) * kb.flatten(prediction))
	union = kb.sum(kb.flatten(prediction))
	return kb.clip((intersection) / (union + 1), 1.0e-8, 1)

def recall(true, prediction):
	prediction = kb.round(prediction)
	intersection = kb.sum(kb.flatten(true) * kb.flatten(prediction))
	union = kb.sum(kb.flatten(true))
	return kb.clip((intersection) / (union + 1), 1.0e-8, 1)

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


def plot(history, model_name, num_epochs, image_size, test=False):
	# Create plot that plots model accuracy vs. epoch

	if test != False:
		path = "./tmp" 
	else:
		path = "./output"
	print("Plotting accuracy")
	fig = plt.figure(figsize=(10, 10))
	plt.plot(history.history['binary_accuracy'])
	plt.plot(history.history['val_binary_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig(path + '/accuracy-' + model_name +"-" + str(num_epochs) +"-" + str(image_size) + '-history1.png'.format(32))
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
	plt.savefig(path + '/f1-' + model_name +"-" + str(num_epochs) + "-" + str(image_size) + '-history1.png'.format(32))
	plt.close(fig)

	print("Finished plotting f1_score")

	fig = plt.figure(figsize=(10, 10))

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig(path + '/loss-' + model_name +"-" + str(num_epochs) + "-" + str(image_size) + '-history1.png'.format(32))
	plt.close(fig)

	print("Finished plotting loss")