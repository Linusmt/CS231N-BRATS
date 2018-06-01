import matplotlib.pyplot as plt
import pickle
import numpy as np
import glob
import csv
import argparse


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

def plot_multiple_models(histories, model_names, num_epochs, image_size, metric="binary_accuracy", save_path="outsput/table_default.csv"):
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
	plt.savefig(save_path + '.png'.format(32))
	print("Finished plotting " + metric )
	plt.close(fig)



IMAGE_SIZE = 64
TEST_MODEL = False
NUM_EPOCHS = 30
METRICS = ["binary_accuracy", "val_binary_accuracy", "brats_f1_score", "val_brats_f1_score", "loss", "val_loss"]
def load_history(model_name):
	with open(model_name, "rb") as history_file:
		return pickle.load( history_file)



def create_table(histories, model_names, save_path="table_default"):
	average_times_per_epochs = []
	metric_arr = [[] for x in METRICS]
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
		data.append([METRICS[i]] + ['%.4f' % x for x in metric_results])

	with open(save_path + ".csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for row in data:
			print (", ".join(row))
			writer.writerow(row)

def main(args):

	make_plots = args.make_plots
	make_table = args.make_table
	image_size = args.image_size
	epochs = args.epochs
	histories = []
	history_files_full_paths = glob.glob("./history/train_history" +"*" + str(epochs) + "*" + str(IMAGE_SIZE) + "*")
	history_files = [x.split("/")[2] for x in history_files_full_paths]
	model_names = ["_".join(x[0:-3].split("_")[2:-1]) for x in history_files]


	for history_file in history_files_full_paths:
		histories.append(load_history(history_file))

	if make_plots:
		print ("\n************************************************")
		print ("Making Plots")
		print ("************************************************\n")

		for metric in METRICS:
			save_path = './merged_graphs/' + metric + "-" + "_".join(MODELS) +"-" + str(epochs) +"-" + str(IMAGE_SIZE) 

			plot_multiple_models(histories, model_names, epochs, image_size, metric, save_path)


	if make_table:
		save_path = './merged_graphs/table_' + "_".join(MODELS) +"-" + str(epochs) +"-" + str(image_size) 

		print ("\n************************************************")
		print ("Making Tables")
		print ("************************************************\n")

		create_table(histories, model_names, save_path)

# print (plot_models(False))

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--epochs', type=int, nargs='?', default=30,
	                    help='number of desired epochs')
	parser.add_argument('--preprocess', type=bool,  default=False,
	                    help='whether to load the dataset again and preprocess')	
	parser.add_argument('--image_size', type=int, nargs='?', default=64,
	                    help='new image size to be chosen')
	parser.add_argument('--make_plots', type=bool, nargs='?', default=True,
	                    help='whether to make_plots')		
	parser.add_argument('--make_table', type=bool, nargs='?', default=True,
	                    help='whether to make a table')	
	parser.add_argument('--lr', type=float, nargs='?', default=1e-3,
	                    help='learning rate as a float')
	parser.add_argument('--use_dropout', type=float, nargs="?", default=0.0,
						help="amount of dropout to use")
	args = parser.parse_args()

	main(args)