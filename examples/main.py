import simflow as sf


def main():
    # load data
    Data, Labels = sf.utils.data_loader_mnist.load_normalized_mnist_data_flat()

    inp_dim = 784
    num_classes = 10
    # create network
    net = sf.model.Model()
    net.add_layer(sf.layers.layers.Dense(inp_dim, 200))
    net.add_layer(sf.layers.activations.ReLU())
    net.add_layer(sf.layers.layers.BN_mean(200))
    net.add_layer(sf.layers.layers.Dense(200, num_classes))

    # add loss function
    net.set_loss_fn(sf.losses.SoftmaxCrossEntropyLoss())

    # add optimizer
    net.set_optimizer(sf.optimizers.SGD(lr=0.01, momentum=0.9, nestrov=True))

    # add iterator
    net.set_iterator(sf.iterators.minibatch_iterator())

    # fit the training data for 5 epochs
    net.fit(Data['train'], Labels['train'], epochs=5)

    # pring scores after training
    print("Final Accuracies after training :")
    print("Train Accuracy: ", net.score(Data['train'],
                                        Labels['train'])[1], end=" ")
    print("validation Accuracy: ", net.score(Data['val'],
                                             Labels['val'])[1], end=' ')
    print("Test Accuracy: ", net.score(Data['test'], Labels['test'])[1])


if __name__ == '__main__':
    main()
