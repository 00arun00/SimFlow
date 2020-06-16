import numpy as np


def load_normalized_mnist_data_flat():
    '''
    Loads and normalizes the MNIST data. Reads the data from

        -data/mnist_train.csv
        -data/mnist_test.csv

    If these are not present they are downloaded and saved from
    https://pjreddie.com/projects/mnist-in-csv/

    Returns:
        :input (dict of numpy.ndarray):  Input
        :labels (dict of numpy.ndarray):  Lables

    Each has keys 'train', 'val', 'test' which map to numpy arrays
    '''
    try:
        print('loading data please wait will take around a minute')
        data = np.loadtxt('data/mnist_train.csv', dtype=int, delimiter=',')
    except OSError:
        import os
        import wget
        url = 'https://pjreddie.com/media/files/mnist_train.csv'
        print('File now found downloading from online source')
        print('Downloading data will take some time')
        try:
            os.mkdir('data')
        except FileExistsError:
            pass
        wget.download(url, 'data/mnist_train.csv')
        print('\nNow loading data please wait will take around a minute')

        data = np.loadtxt('data/mnist_train.csv', dtype=int, delimiter=',')

    try:
        print('loading data please wait will take around a minute')
        test_data = np.loadtxt('data/mnist_test.csv', dtype=int, delimiter=',')
    except OSError:
        import os
        import wget
        url = 'https://pjreddie.com/media/files/mnist_test.csv'
        print('File now found downloading from online source')
        print('Downloading data will take some time')
        try:
            os.mkdir('data')
        except FileExistsError:
            pass
        wget.download(url, 'data/mnist_test.csv')
        print('\nNow loading data please wait will take around a minute')

        test_data = np.loadtxt('data/mnist_test.csv', dtype=int, delimiter=',')

    inputs = dict()
    labels = dict()

    train_data = data[:50000]
    train_inputs = train_data[:, 1:]

    val_data = data[50000:]
    val_inputs = val_data[:, 1:]

    test_inputs = test_data[:, 1:]

    mean = np.mean(train_inputs)
    std = np.std(train_inputs)

    inputs['train'] = (train_inputs - mean)/std
    inputs['val'] = (val_inputs - mean)/std
    inputs['test'] = (test_inputs - mean)/std

    labels['train'] = train_data[:, 0]
    labels['val'] = val_data[:, 0]
    labels['test'] = test_data[:, 0]
    print('Data loading completed')
    return inputs, labels


def load_normalized_mnist_data_conv():
    '''
    Loads and normalizes the MNIST data. Reads the data from

        -data/mnist_train.csv
        -data/mnist_test.csv

    If these are not present they are downloaded and saved from
    https://pjreddie.com/projects/mnist-in-csv/

    Returns:
        :input (dict of numpy.ndarray):  Input
        :labels (dict of numpy.ndarray):  Lables

    Each has keys 'train', 'val', 'test' which map to numpy arrays
    '''
    print('loading data please wait will take around a minute')
    data = np.loadtxt('data/mnist_train.csv', dtype=int, delimiter=',')
    test_data = np.loadtxt('data/mnist_test.csv', dtype=int, delimiter=',')
    inputs = dict()
    labels = dict()

    train_data = data[:50000]
    train_inputs = train_data[:, 1:].reshape(-1, 1, 28, 28)

    val_data = data[50000:]
    val_inputs = val_data[:, 1:].reshape(-1, 1, 28, 28)

    test_inputs = test_data[:, 1:].reshape(-1, 1, 28, 28)

    mean = np.mean(train_inputs)
    std = np.std(train_inputs)

    inputs['train'] = (train_inputs - mean)/std
    inputs['val'] = (val_inputs - mean)/std
    inputs['test'] = (test_inputs - mean)/std

    labels['train'] = train_data[:, 0]
    labels['val'] = val_data[:, 0]
    labels['test'] = test_data[:, 0]
    print('Data Loading completed')
    return inputs, labels
