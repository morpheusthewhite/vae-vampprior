import numpy as np

import os
import urllib3
import pickle
import tensorflow as tf


DATASETS_FOLDER = "datasets"
FREY_URL = "https://raw.githubusercontent.com/y0ast/Variational-Autoencoder/master/freyfaces.pkl"


def load_frey(train_samples=1500, MB=100):
    assert train_samples % MB == 0

    frey_path = os.path.join(DATASETS_FOLDER, "freyfaces.pkl")

    if not os.path.exists(frey_path):
        print('Downloading dataset...')
        http = urllib3.PoolManager()
        r = http.request('GET', FREY_URL)

        # create directory if does not exist
        if not os.path.exists(DATASETS_FOLDER):
            os.mkdir(DATASETS_FOLDER)

        # saving pkl file
        with open(frey_path, 'wb') as f:
            f.write(r.data)

    # opening pkl file, containing np array
    with open(frey_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')  # only 1965 observations

    # array is reshaped in order for array[n] to be a rectangular image
    data = np.reshape(data, (-1, 28, 20))
    np.random.shuffle(data)

    test_samples = data.shape[0] - train_samples
    # discard samples to make test_samples divisible by the number of elements
    # in a minibatch
    test_samples -= test_samples % MB

    x_train, x_test = data[:train_samples], data[-test_samples:]

    return np.array(x_train, dtype=np.float32), np.array(x_test, dtype=np.float32)


def load_fashion_mnist():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, _), (x_test, _) = fashion_mnist.load_data()
    x_train = np.array(x_train / 255., dtype=np.float32)
    x_test = np.array(x_test / 255., dtype=np.float32)

    return x_train, x_test
