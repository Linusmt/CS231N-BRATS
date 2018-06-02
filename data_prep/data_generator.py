import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, y, n):
        'Initialization'
        self.batch_size = 1
        self.X = X
        self.y = y
        self.n = n



    def __len__(self):
        'Denotes the number of batches per epoch'
        return 400

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        index = np.random.randint(0, self.n)

        # Get the original data
        X =np.expand_dims(self.X[index], 0)
        y = np.array([self.y[index]])

        # Find list of IDs
        mod_random = np.random.uniform(0,1)

        if np.random.uniform(0,1) < 0.3:
            X = np.flip(X, axis=1)
            y = np.flip(y, axis=1)

        if np.random.uniform(0,1) < 0.3:

            X = np.flip(X, axis=2)
            y = np.flip(y, axis=2)

        if np.random.uniform(0,1) < 0.3:
            X = np.flip(X, axis=3)
            y = np.flip(y, axis=3)
        # Generate data
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