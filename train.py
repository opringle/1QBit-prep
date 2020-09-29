import logging
import argparse

from src import model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model for ..."
    )
    group = parser.add_argument_group('Computation args')
    group.add_argument('--gpus', type=int, default=0, help='Number of GPUs to train on. 0 for CPU.')
    return parser.parse_args()


def train():
    """
    Fit a neural network to training data and save parameters to disk
    """
    net = model.Classifier()
    pass


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    train()
