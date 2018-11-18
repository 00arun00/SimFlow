# SimFlow
Ultra portable Deep Learning framework in Numpy

### Currently supported features

#### Layers:

  - Dense
  - ReLU
  - Sigmoid
  - Tanh
  - BN_mean (mean only batch norm)

#### Losses:

  - SoftmaxCrossEntropyLoss

#### Optimizers:

  - SGD
  - Momentum
  - Nestrov Momentum

#### Iterators:

  - Full batch
  - Mini batch
  - Stochastic

#### Data Loaders:

  - MNIST



### Installation steps:

```
pip install -r requirements.txt
```





### Sample network/ How to use

```python
Data,Labels = sf.data_loader_mnist.load_normalized_mnist_data_flat()

inp_dim = 784
num_classes = 10

#create network
net = sf.Model()
net.add_layer(sf.layers.Dense(inp_dim,200))
net.add_layer(sf.layers.ReLU())
net.add_layer(sf.layers.BN_mean(200))
net.add_layer(sf.layers.Dense(200,num_classes))

#add loss function
net.set_loss_fn(sf.losses.SoftmaxCrossEntropyLoss())

# add optimizer
net.set_optimizer(sf.optimizers.SGD(lr=0.01,momentum=0.9,nestrov=True))

# add iterator
net.set_iterator(sf.iterators.minibatch_iterator())

# fit the training data for 5 epochs
net.fit(Data['train'],Labels['train'],epochs=5)

# pring scores after training
    print("Final Accuracies after training :")
    print("Train Accuracy: ",net.score(Data['train'],Labels['train'])[1],end=" ")
    print("validation Accuracy: ",net.score(Data['val'],Labels['val'])[1],end =' ')
    print("Test Accuracy: ",net.score(Data['test'],Labels['test'])[1])

```

### Features currently worked on:

#### Layers:

- Convolutional Neural nets
- Dilated Convolutional Layer
- Batch Normalization
- dropout
- maxpool / average pool
- LeakyReLU
- PReLU



#### Regularizers:

- L1 
- L2
- elastic net

#### Optimizers

- Adam
- RMSprop



### Testing Features

run the following command to check if all your layers are functional

```
python -m pytest -v
```

currently supports 

- Dense Layer
- BN_mean Layer

