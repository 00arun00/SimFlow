import numpy as np
class Iterator(object):
    def __init__(self,batch_size=128,*,shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.method = 'direct'
    def get_iterator(self,Data,Labels):
        n_train = Data.shape[0]
        assert Data.shape[0]==Labels.shape[0],'Number of Data not same as number of Labels'
        if self.method == 'full batch':
            self.batch_size = n_train
        if self.shuffle:
            order = np.random.permutation(n_train)
        else:
            order = np.arange(n_train)
        start_idx = 0
        while start_idx < n_train:
            end_idx = min(start_idx+self.batch_size, n_train)
            idxs = order[start_idx:end_idx]
            mb_inputs = Data[idxs]
            mb_labels = Labels[idxs]
            yield mb_inputs,mb_labels
            start_idx += self.batch_size
    def __repr__(self):
        return f'{self.method} iterator with shuffling = {self.shuffle}'

class minibatch_iterator(Iterator):
    def __init__(self,batch_size=128,*,shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.method = 'mini_batch'
    def __repr__(self):
        return f'mini_batch iterator with batch_size = {self.batch_size} and shuffling = {self.shuffle}'

class fullbatch_iterator(Iterator):
    def __init__(self,*,shuffle=True):
        self.batch_size = -1
        self.shuffle = shuffle
        self.method = 'full batch'

class stochastic_iterator(Iterator):
    def __init__(self,*,shuffle=True):
        self.batch_size=1
        self.method = 'stochastic'
        self.shuffle = shuffle
