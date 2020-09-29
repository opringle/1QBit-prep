import logging
import argparse
import os
import multiprocessing
import numpy as np
import mxnet as mx
from mxnet import nd, gluon, autograd

from src import model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model for ..."
    )
    # Computation
    group = parser.add_argument_group('Computation args')
    group.add_argument('--gpus', type=int, default=0, help='Number of GPUs to train on. 0 for CPU.')

    # Optimizer
    group = parser.add_argument_group('Optimization arguments')
    group.add_argument('--epochs', type=int, default=10,
                       help='num of times to loop through training data')
    return parser.parse_args()


def evaluate_accuracy(data_iterator, net, ctx):
    """
    :param data_iterator: gluon data loader
    :param net: gluon hybrid sequential block
    :return: network accuracy on data
    """
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


def train():
    """
    Fit a neural network to training data and save parameters to disk
    """
    """
        train function formatted for use with amazon sagemaker (hence unused arguments)
        :param hyperparameters: dict of network hyperparams
        :param channel_input_dirs: dict of paths to train and val data
        :param num_gpus: number of gpus to distribute training on (0 for cpu)
        :return: gluon neural network with learned network parameters
        """
    ctx = mx.gpu() if args.gpus > 0 else mx.cpu()
    logging.info("Training context: {}".format(ctx))
    BATCH_SIZE = 32

    logging.info("Downloading the MNIST dataset, building train/test data loaders")
    train_data = mx.gluon.data.DataLoader(
        mx.gluon.data.vision.MNIST(train=True, transform=lambda data, label: (data.astype(np.float32) / 255, label)),
        batch_size=BATCH_SIZE,
        num_workers=multiprocessing.cpu_count(),
        shuffle=True)
    test_data = mx.gluon.data.DataLoader(
        mx.gluon.data.vision.MNIST(train=False, transform=lambda data, label: (data.astype(np.float32) / 255, label)),
        batch_size=BATCH_SIZE,
        num_workers=multiprocessing.cpu_count(),
        shuffle=False)
    logging.info("{} train records and {} test records (after padding last batch)".
                 format(len(train_data) * BATCH_SIZE, len(test_data) * BATCH_SIZE))

    net = model.CnnClassifier(dropout=0.2, num_label=10)
    logging.info("Network architecture: {}".format(net))

    logging.info("Hybridizing network to convert from imperitive to symbolic for increased training speed")
    net.hybridize()

    logging.info("Initializing network parameters with Xavier Algorithm")
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

    optimizer = gluon.Trainer(params=net.collect_params(), optimizer='sgd',
                              optimizer_params={'learning_rate': 0.001,
                                                'momentum': 0.9,
                                                'wd': 0.0})

    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    sm_loss = gluon.loss.SoftmaxCrossEntropyLoss()  # softmax function followed by cross entropy loss
    epochs = args.epochs
    logging.info("Training for {} epochs...".format(epochs))
    for e in range(epochs):
        epoch_loss = 0
        weight_updates = 0
        for data, label in train_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                pred = net(data)
                loss = sm_loss(pred, label)
            loss.backward()
            optimizer.step(data.shape[0])
            epoch_loss += nd.mean(loss).asscalar()
            weight_updates += 1
        train_accuracy = evaluate_accuracy(train_data, net, ctx)
        val_accuracy = evaluate_accuracy(test_data, net, ctx)
        net.save_parameters(os.path.join('checkpoint', 'epoch' + str(e) + '.params'))
        logging.info("Epoch{}: Average Record Train Loss= {:.4} Train Accuracy= {:.4} Validation Accuracy= {:.4}".
                     format(e, epoch_loss / weight_updates, train_accuracy, val_accuracy))
    return net


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    train()
