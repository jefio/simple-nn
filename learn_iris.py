"""Learn a neural network on the iris dataset and evaluate its accuracy."""

import argparse
import logging

import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from neural_network import NeuralNetwork


logger = logging.getLogger(__name__)
# reproducible results
np.random.seed(12345)


def get_iris_data(train_ratio):
    iris = datasets.load_iris()
    x, y = iris.data, iris.target
    classes = set(y)
    logger.info("x=%s, classes=%s", x.shape, classes)
    x = preprocessing.scale(x)
    logger.info("mean=%s, std=%s", x.mean(axis=0), x.std(axis=0))
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)
    logger.info("x_train=%s, x_val=%s, x_test=%s", x_train.shape, x_val.shape, x_test.shape)
    return [
        x_train, x_val, x_test, one_hot(y_train, len(classes)),
        one_hot(y_val, len(classes)), one_hot(y_test, len(classes))]


def one_hot(y, n_classes):
    """
    Params
    ------
    y : ndarray
        y[i] = class of sample i

    Returns
    -------
    ndarray
        m[i, c] = 1 if i belongs to class c, 0 otherwise
    """
    n_sample = y.shape[0]
    m = np.zeros((n_sample, n_classes), int)
    m[range(n_sample), y] = 1
    return m


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', type=int, nargs='+', default=[8], help='Sizes of the hidden layers')
    parser.add_argument('--l2-reg', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--learning-rate', type=float, default=1)
    parser.add_argument('--loss-name', type=str, choices=('squared_loss', 'binary_cross_entropy'), default='squared_loss')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--early-stopping', type=int, default=0, choices=(0, 1))
    parser.add_argument('--train-ratio', type=float, default=0.6,
                        help='Ratio for training data (remaining data will be used for validation + test)')
    args = parser.parse_args()

    x_train, x_val, x_test, y_train, y_val, y_test = get_iris_data(args.train_ratio)
    # iris has 4 features and 3 classes
    sizes = [4] + args.sizes + [3]
    nn = NeuralNetwork(
        sizes, l2_reg=args.l2_reg,
        dropout=args.dropout, learning_rate=args.learning_rate,
        loss_name=args.loss_name,
        early_stopping=args.early_stopping)
    nn.fit(
        x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(x_val, y_val))
    nn.plot_statistics()
    test_acc = nn.get_accuracy(x_test, y_test)
    logger.info("test_acc=%s", test_acc)


if __name__ == '__main__':
    main()
