import matplotlib.pyplot as plt
import pickle
import numpy as np
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


def plot_multiple_models(histories, model_names, num_epochs, image_size, metric="binary_accuracy"):
	# Create plot that plots model accuracy vs. epoch
	fig = plt.figure(figsize=(10, 10))

	for i in range(len(histories)):
		history = histories[i]
		epochs = np.arange(len(history)) + 1
		model_name = model_names[i]
		print("Plotting: " + metric + " for model " + model_name )
		print("Max_val", np.max(history[metric]), "  Min_val: ", np.min(history[metric])) 

		plt.plot( history[metric])
		# plt.plot(history['val_binary_accuracy'])
	plt.title(model_name + " " + metric)
	plt.ylabel(metric)
	plt.xlabel('epoch')
	plt.legend(model_names, loc='upper left')
	# plt.show()
	plt.savefig('./output/' + metric + "-" + "_".join(model_names) +"-" + str(num_epochs) +"-" + str(image_size) + '-history1.png'.format(32))
	print("Finished plotting " + metric )
	plt.close(fig)


MODELS = [ "baseline", "u3d", "u3d_inception" ]
IMAGE_SIZE = 64
TEST_MODEL = False
NUM_EPOCHS = 15
METRICS = ["binary_accuracy", "val_binary_accuracy", "brats_f1_score", "val_brats_f1_score", "loss", "val_loss"]
def load_history(model_name, num_epochs, image_size, test_model):
	#Save the model training history for later inspection
	npy_file_name = "_".join(['./train_history', model_name, str(num_epochs), str(image_size),"test" if test_model else ""])+".pkl"
	print (npy_file_name)
	with open(npy_file_name, "rb") as history_file:
		return pickle.load( history_file)

def plot_models():
	histories = []
	for model_name in MODELS:
		histories.append(load_history(model_name, NUM_EPOCHS, IMAGE_SIZE, TEST_MODEL))
	# print (histories)
	for metric in METRICS:
		plot_multiple_models(histories, MODELS, NUM_EPOCHS, IMAGE_SIZE, metric)


print (plot_models())