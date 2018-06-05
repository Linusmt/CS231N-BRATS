import numpy as np
import keras
import glob

# LABELS_PATH = 
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_paths, label_paths,test=False):
        'Initialization'
        self.batch_size = 1
        self.data_paths = data_paths#glob.glob(data_path +"/*.npy")
        self.label_paths = label_paths#glob.glob(label_path +"/*.npy")
        # self.h = X
        # self.y = y
        self.num_batch = 20 if test else 1000
        self.n = len(self.data_paths)



    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.num_batch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        index = np.random.randint(0, self.n)
        X = np.array([np.load(self.data_paths[index])])
        y = np.array([np.load(self.label_paths[index])])
        # print(np.sum(y==4))
        # y[y != 4] = 0
        # y[y != 1] = 0
        # print(y.shape, np.sum(y))
        # # Get the original data
        # X =np.expand_dims(self.X[index], 0)
        # y = np.array([self.y[index]])

        # # Find list of IDs
        # mod_random = np.random.uniform(0,1)

        ####PROBLEM IS HERE
        ### Need to properly create the y data 

        y_true = np.zeros(list(y.shape) + [5])
        for i in range(5):
            y_true[y == i,i] = 1

        y = y_true
        # if np.random.uniform(0,1) < 0.15:
        #     X = np.flip(X, axis=1)
        #     y = np.flip(y, axis=1)

        # if np.random.uniform(0,1) < 0.15:

        #     X = np.flip(X, axis=2)
        #     y = np.flip(y, axis=2)

        # if np.random.uniform(0,1) < 0.15:
        #     X = np.flip(X, axis=3)
        #     y = np.flip(y, axis=3)
        # # Generate data
        # X, y = self.__data_generation(list_IDs_temp)

        return X, y

class ValidationDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_paths, label_paths, test=False):
        'Initialization'
        self.batch_size = 1
        self.data_paths = data_paths#glob.glob(data_path +"/*.npy")
        self.label_paths = label_paths#glob.glob(label_path +"/*.npy")
        # self.h = X
        # self.y = y
        self.n = len(self.data_paths)
        self.index = 0
        self.num_batch = 20 if test else 200



    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.num_batch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        X = np.array([np.load(self.data_paths[self.index])])
        y = np.array([np.load(self.label_paths[self.index])])

        y_true = np.zeros(list(y.shape) + [5])
        for i in range(5):
            y_true[y == , i ] = 1

        y = y_true
        # y[y != 4] = 0
        # y[y != 1] = 0
        self.index += 1
        # # Get the original data
        # X =self.X[index], 0)
        # y = np.array([self.y[index]])

        # # Find list of IDs
        # mod_random = np.random.uniform(0,1)

        # if np.random.uniform(0,1) < 0.3:
        #     X = np.flip(X, axis=1)
        #     y = np.flip(y, axis=1)

        # if np.random.uniform(0,1) < 0.3:

        #     X = np.flip(X, axis=2)
        #     y = np.flip(y, axis=2)

        # if np.random.uniform(0,1) < 0.3:
        #     X = np.flip(X, axis=3)
        #     y = np.flip(y, axis=3)
        # # Generate data
        # X, y = self.__data_generation(list_IDs_temp)

        return X, y
    # def on_epoch_end(self):
    #     'Updates indexes after each epoch'
    #     self.indexes = np.arange(len(self.list_IDs))
    #     if self.shuffle == True:
    #         np.random.shuffle(self.indexes)

    # def __data_generation(self, list_IDs_temp):
    #     'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    #     # Initialization
    #     X = np.empty((self.batch_size, *self.dim, self.n_channels))
    #     y = np.empty((self.batch_size), dtype=int)

    #     # Generate data
    #     for i, ID in enumerate(list_IDs_temp):
    #         # Store sample
    #         X[i,] = np.load('data/' + ID + '.npy')

    #         # Store class
    #         y[i] = self.labels[ID]

    #     return X, keras.utils.to_categorical(y, num_classes=self.n_classes)