import matplotlib.pyplot as plt
import pickle
import numpy as np
import glob
import csv

def plot(history, model_name, num_epochs, image_size):
	# Create plot that plots model accuracy vs. epoch and save to the output directory

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

MODELS = [ "baseline", "u3d", "u3d_inception", "ures", "use","use_res" ]

def plot_multiple_models(histories, model_names, num_epochs, image_size, metric="binary_accuracy"):
	# Create plot that plots model accuracy vs. epoch
	fig = plt.figure(figsize=(10, 10))

	for i in range(len(histories)):
		history = histories[i]
		epochs = np.arange(len(history)) + 1
		model_name = model_names[i]
		# print("Plotting: " + metric + " for model " + model_name )
		# print("Max_val", np.max(history[metric]), "  Min_val: ", np.min(history[metric])) 

		plt.plot( history[metric])
		# plt.plot(history['val_binary_accuracy'])
	plt.title(model_name + " " + metric)
	plt.ylabel(metric)
	plt.xlabel('epoch')
	plt.legend(model_names, loc='upper left')
	# plt.show()
	plt.savefig('./output/' + metric + "-" + "_".join(MODELS) +"-" + str(num_epochs) +"-" + str(image_size) + '-history1.png'.format(32))
	print("Finished plotting " + metric )
	plt.close(fig)



IMAGE_SIZE = 64
TEST_MODEL = False
NUM_EPOCHS = 30
METRICS = ["binary_accuracy", "val_binary_accuracy", "brats_f1_score", "val_brats_f1_score", "loss", "val_loss"]
def load_history(model_name):
	with open(model_name, "rb") as history_file:
		return pickle.load( history_file)



def make_table(histories, model_names):
	average_times_per_epochs = []
	metric_arr = [[] for x in METRICS]
	print (metric_arr, len(METRICS))
	for i in range(len(histories)):
		history = histories[i]
		model_name = model_names[i]

		average_times_per_epochs.append(np.mean(history["times"]))
		best_model = np.argmax(history["val_brats_f1_score"])

		for j in range(len(METRICS)):
			metric = METRICS[j]
			metric_arr[j].append(history[metric][best_model])



	data = []

	data.append(["model"]+model_names)
	data.append(["average_time_per_epoch"] + ['%.4f' % x for x in average_times_per_epochs])

	for i in range(len(metric_arr)):

		metric_results = metric_arr[i]
		# print(METRICS[i])
		data.append([METRICS[i]] + ['%.4f' % x for x in metric_results])
		# print (  ", ".join(['%.4f' % x for x in metric_results]))
	stringified_data = [",".join(x) for x in data]

	with open("tables_temp", "w") as csvfile:
		writer = csv.writer(csvfile)
		for row in data:
			print (", ".join(row))
			writer.writerow(row)

def plot_models(make_plots=True, create_table=True):
	histories = []
	history_files_full_paths = glob.glob("./history/train_history" +"*" + str(NUM_EPOCHS) + "*" + str(IMAGE_SIZE) + "*")
	history_files = [x.split("/")[2] for x in history_files_full_paths]
	model_names = ["_".join(x[0:-3].split("_")[2:-1]) for x in history_files]

	for history_file in history_files_full_paths:
		histories.append(load_history(history_file))

	if make_plots:
		for metric in METRICS:
			plot_multiple_models(histories, model_names, NUM_EPOCHS, IMAGE_SIZE, metric)

	if create_table:
		make_table(histories, model_names)

print (plot_models(False))

