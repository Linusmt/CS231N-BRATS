import pickle
# from utils import plot
import matplotlib.pyplot as plt


def plot(histories, model_names):
	# Create plot that plots model accuracy vs. epoch
	# print("Plotting accuracy")
	# fig = plt.figure(figsize=(10, 10))
	# plt.plot(history['binary_accuracy'])
	# plt.plot(history['val_binary_accuracy'])
	# plt.title('model accuracy')
	# plt.ylabel('accuracy')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'validation'], loc='upper left')
	# plt.show()
	# print("Finished plotting accuracy")
	# plt.close(fig)
	# print("Plotting f1_score")

	fig = plt.figure(figsize=(10, 10))

	# plt.plot(history['brats_f1_score'])
	for history in histories:
		plt.plot(history['val_brats_f1_score'])

	plt.title('Validation F1 score')
	plt.ylabel('f1 score')
	plt.xlabel('epoch')
	plt.legend(model_names, loc='upper left')
	plt.savefig('dev_f1.png'.format(32))

	plt.show()
	plt.close(fig)


	fig = plt.figure(figsize=(10, 10))

	# plt.plot(history['brats_f1_score'])
	for history in histories:
		plt.plot(history['brats_f1_score'])

	plt.title('Train F1 score')
	plt.ylabel('f1 score')
	plt.xlabel('epoch')
	plt.legend(model_names, loc='upper left')
	plt.savefig('train_f1.png'.format(32))

	plt.show()
	plt.close(fig)


	print("Finished plotting f1_score")

	fig = plt.figure(figsize=(10, 10))
	for history in histories:
		plt.plot(history['loss'])
		
	# plt.plot(history['loss'])
	# plt.plot(history['val_loss'])
	plt.title('Training loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(model_names, loc='upper left')
	plt.savefig('dev_loss.png'.format(32))

	plt.show()
	plt.close(fig)

	print("Finished plotting loss")

	fig = plt.figure(figsize=(10, 10))
	for history in histories:
		plt.plot(history['val_loss'])
		
	# plt.plot(history['loss'])
	# plt.plot(history['val_loss'])
	plt.title('Training loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(model_names, loc='upper left')
	plt.savefig('train_loss.png'.format(32))
	plt.show()
	plt.close(fig)

	print("Finished plotting loss")



model_names = ["baseline_30_64_notaug", "u3d_inception_30_64_aug", "u3d_inception_30_64_notaug","u3d_30_64_aug", "u3d_30_64_notaug"]
histories = []
for model_name in model_names:
	f = open("train_history_" + model_name + ".pkl", "rb")
	history = pickle.load(f)
	histories.append(history)
plot(histories, model_names)
