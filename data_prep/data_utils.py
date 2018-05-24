import data_prep.utils as utils
import pickle

MRI_PATH = 'baseline_mris'
LABELS_PATH = 'baseline_labels'	

MRI_LOAD_PATH = './data/HGG/**/*Flair*.mha'
LABELS_LOAD_PATH = './data/HGG/**/*seg*.nii'

def load_data(image_size, preprocess, augment_data=False):
	print (image_size)
	if preprocess:
		print ("Preprocessing Data Set")
		utils.preproc_brats_data(MRI_LOAD_PATH, LABELS_LOAD_PATH, image_size, 'baseline', save=True)

	#Paths for the MRI data for the given image size 

	mris, labels = utils.get_brats_data(MRI_PATH, LABELS_PATH, image_size, 'baseline', save=False, preprocess=True, shuffle=False)
	validation_set = (mris[:20,], labels[:20,])
	if augment_data:
		mris, labels  = utils.augment_data(mris[20:,], labels[20:,])
	else:
		mris, labels  = mris[20:,], labels[20:,]
	return mris, labels, validation_set

