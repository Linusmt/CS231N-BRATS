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
import pickle


MRI_FLAIR_LOAD_PATH = './data/HGG/**/*flair*.nii'
MRI_T1c_PATH = './data/HGG/**/*t1c*.nii'
MRI_T2_LOAD_PATH = './data/HGG/**/*t2*.nii'
DATA_PATHS = [MRI_T1c_PATH, MRI_T2_LOAD_PATH, MRI_FLAIR_LOAD_PATH]
LABELS_LOAD_PATH = './data/HGG/**/*seg*.nii'
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

def take_data_crops(image, labels, crop_size=64, num_crops=5):
	h, w, d, c = image.shape
	heights = np.random.randint(0, h-crop_size, size=num_crops)
	widths = np.random.randint(0, w-crop_size, size=num_crops)
	depths = np.random.randint(0, d-crop_size, size=num_crops)

	image_crops = []
	labels_crops = []
	for i in range(num_crops):
		image_crops.append(image[heights[i]:heights[i]+crop_size,widths[i]:widths[i]+crop_size,depths[i]:depths[i]+crop_size,:])
		labels_crops.append(labels[heights[i]:heights[i]+crop_size,widths[i]:widths[i]+crop_size,depths[i]:depths[i]+crop_size])

	return image_crops, labels_crops

# def take_label_crop(label):

# def brats_preprocess_mri(images, newsize, name='unknown', save=False):
# 	preprocess = [ resize_by_slice_mri((i - i.mean()) / i.std(), newsize) for i in images ]
# 	return np.array(preprocess)[..., np.newaxis].astype('float32')

# def brats_preprocess_labels(images, newsize, name='unknown', save=False):
# 	preprocessed_labels = []
# 	for i in images:
# 		i[i != 4] = 0
# 		i[i == 4] = 1
# 		preprocessed_labels.append(resize_by_slice_label(i, newsize))
# 	if save:
# 		if name == 'unknown': print('Cannot save file unless new filename is specified')
# 		else: np.save(name, np.array(preprocessed_labels)[..., np.newaxis].astype('float32'))
# 	return np.array(preprocessed_labels)[..., np.newaxis].astype('float32')

# def brats_load_data(path, dataname='unknown', preprocess=False):
# 	print('Loading ' + dataname + ' data...')
# 	# print( glob.glob(path, recursive=True))
# 	if not preprocess:
# 		return [ nib.load(f).get_data() for f in glob.glob(path, recursive=True) ]
# 	else:
# 		return np.load(path)

# def brats_preprocess_mri(images, newsize, name='unknown', save=False):
# 	preprocess = [ resize_by_slice_mri((i - i.mean()) / i.std(), newsize) for i in images ]
# 	return np.array(preprocess)[..., np.newaxis].astype('float32')

# def brats_preprocess_labels(images, newsize, name='unknown', save=False):
# 	preprocessed_labels = []
# 	for i in images:
# 		i[i != 4] = 0
# 		i[i == 4] = 1
# 		preprocessed_labels.append(resize_by_slice_label(i, newsize))
# 	if save:
# 		if name == 'unknown': print('Cannot save file unless new filename is specified')
# 		else: np.save(name, np.array(preprocessed_labels)[..., np.newaxis].astype('float32'))
# 	return np.array(preprocessed_labels)[..., np.newaxis].astype('float32')
def brats_crop_item_preprocess(flair_path, t2_path, t1c_path,labels_path, image_save_path, labels_save_path,crop_size=64, num_crops=5):
	# print('Loading ' + dataname + ' data...')
	# print( glob.glob(path, recursive=True))
	flair = nib.load(flair_path).get_data()
	t2 = nib.load(t2_path).get_data()
	t1c = nib.load(t1c_path).get_data()
	labels = nib.load(labels_path).get_data()
	image = np.stack([flair, t2, t1c], axis=3)

	x_crops, y_crops = take_data_crops(image, labels, num_crops=num_crops, crop_size=crop_size)
	path = flair_path.split("/")
	print(path)

	for crop in range(num_crops):
		np.save(image_save_path +"_crop_" +  str(crop + 1), np.array(x_crops[crop]).astype('float32'))
		np.save(labels_save_path + "_label_" + str(crop + 1), np.array(y_crops[crop]).astype('float32'))

	# if not preprocess:
	# 	return nib.load(path).get_data() 
	# else:
	# 	return np.load(path)
# LABELS_LOAD_PATH = './data/HGG/**/*seg*.nii'

def preproc_brats_data(image_size, model_name='unknown',  save=False, preprocess=False, num_crops=10):
	channels = []
	flair_paths = glob.glob(MRI_FLAIR_LOAD_PATH, recursive=True)
	t2_paths = glob.glob(MRI_T2_LOAD_PATH, recursive=True)
	t1c_paths = glob.glob(MRI_T1c_PATH, recursive=True)
	labels_paths = glob.glob(LABELS_LOAD_PATH, recursive=True)
	for i in range(len(flair_paths)):
		images_save_path = "./data/crops/" + flair_paths[i].split("/")[3] 
		labels_save_path = "./data/labels/" + flair_paths[i].split("/")[3] 
		print(len(flair_paths), len(t2_paths), len(t1c_paths), len(labels_paths))
		brats_crop_item_preprocess(flair_paths[i], t2_paths[i], t1c_paths[i], labels_paths[i],images_save_path, labels_save_path, crop_size=image_size, num_crops=10)
		# mris = brats_preprocess_mri(mris, image_size, model_name + '_mris_' +str(image_size), save)

	# print(mris.shape)
	# np.save(model_name + '_mris_' +str(image_size), np.array(mris).astype('float32'))


	# labels = brats_crop_item_preprocess(labels_path, 'labels', preprocess)
	# labels = brats_preprocess_labels(labels, image_size, model_name + '_labels_' +str(image_size), save)


def get_brats_data(mri_path, labels_path, image_size, model_name='unknown', preprocess=False, save=False, shuffle=True):
	
	full_mri_path = mri_path + "_" + str(image_size) + ".npy"
	full_labels_path = labels_path + "_" + str(image_size) + ".npy"

	# Load data
	mris = np.load(full_mri_path)
	labels =  np.load(full_labels_path)


	# Return data and labels
	return mris, labels

def augment_data(X, Y):
	new_X = [X, np.flip(np.flip(X, axis=1), axis=2)]#, np.flip(X, axis=2), np.flip(X, axis=3)]
	new_Y = [Y, np.flip(np.flip(Y, axis=1), axis=2)]#, np.flip(Y, axis=2), np.flip(Y, axis=3)]
	new_X = np.concatenate(new_X, axis=0)
	new_Y = np.concatenate(new_Y, axis=0)
	return new_X,new_Y


MRI_PATH = 'baseline_mris'
LABELS_PATH = 'baseline_labels'	

MRI_LOAD_PATH = './data/HGG/**/*Flair*.mha'
LABELS_LOAD_PATH = './data/HGG/**/*seg*.nii'

def load_data(image_size, preprocess, augment_data=False, num_crops=10):
	print (image_size)
	if preprocess:
		print ("Preprocessing Data Set")
		preproc_brats_data(image_size, 'baseline', save=True, num_crops=10)

	#Paths for the MRI data for the given image size 

	mris, labels = get_brats_data(MRI_PATH, LABELS_PATH, image_size, 'baseline', save=False, preprocess=True, shuffle=False)
	validation_set = (mris[:20,], labels[:20,])
	if augment_data:
		mris, labels  = augment_data(mris[20:,], labels[20:,])
	else:
		mris, labels  = mris[20:,], labels[20:,]
	return mris, labels, validation_set

