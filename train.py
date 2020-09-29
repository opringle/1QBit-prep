import logging
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model for ..."
    )
    group = parser.add_argument_group('Computation args')
    group.add_argument('--gpus', type=int, default=0, help='Number of GPUs to train on. 0 for CPU.')
    return parser.parse_args()


def train(num_gpus: int):
    pass


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    train(num_gpus=args.gpus)
