"""Feedforward Neural Network"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
import pandas as pd


logger = logging.getLogger(name=__name__)


class NeuralNetwork(object):
    def __init__(self, sizes, l2_reg=0, dropout=0, learning_rate=1,
                 init_method='randn', loss_name='squared_loss',
                 early_stopping=True):
        """
        Params
        ------
        sizes : list
            [size of input layer, ..., size of last layer]
        l2_reg : float
            L2-norm regularization coefficient >= 0
        dropout : float
            Probability of a unit being killed
        """
        self.sizes = sizes
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.loss_name = loss_name
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping

        self.weights = self._get_initial_weights(init_method)
        self.biases = self._get_initial_biases()
        self.training_losses = []
        self.val_accs = []

    def _get_initial_weights(self, init_method):
        """Initialize the network's weights"""
        if init_method == 'randn':
            weights = [
                nr.randn(self.sizes[idx - 1], size)
                for idx, size in enumerate(self.sizes)
                if idx - 1 >= 0]
        else:
            raise ValueError("init_method invalid")
        return weights

    def _get_initial_biases(self):
        """Initialize the network's biases"""
        # no bias for the input layer
        biases = [np.zeros(size) for size in self.sizes[1:]]
        return biases

    def get_losses(self, y_true, y_pred):
        """
        Params
        ------
        y_true : ndarray
            (n_sample, n_classes)
        y_pred : ndarray
            (n_sample, n_classes)

        Returns
        ------
        ndarray
            (n_sample,)
        """
        n_sample, _ = y_true.shape
        assert y_true.shape == y_pred.shape
        if self.loss_name == 'squared_loss':
            losses = squared_loss(y_true, y_pred)
        elif self.loss_name == 'binary_cross_entropy':
            losses = binary_cross_entropy(y_true, y_pred)
        else:
            raise ValueError("loss_name invalid")
        assert losses.shape == (n_sample,)
        return losses

    def get_der_losses(self, y_true, y_pred):
        """
        Params
        ------
        y_true : ndarray
            (n_sample, n_classes)
        y_pred : ndarray
            (n_sample, n_classes)

        Returns
        ------
        ndarray
            (n_sample, n_classes)
        """
        n_sample, _ = y_true.shape
        assert y_true.shape == y_pred.shape
        if self.loss_name == 'squared_loss':
            der_losses = der_squared_loss(y_true, y_pred)
        elif self.loss_name == 'binary_cross_entropy':
            der_losses = der_binary_crossentropy(y_true, y_pred)
        else:
            raise ValueError("loss_name invalid")
        assert der_losses.shape == y_true.shape
        return der_losses

    def _forward_pass(self, x_train, y_train, dropout_layer=None):
        """Compute the weighted inputs z and activation outputs for each neuron,
        compute the final loss

        Parameters
        ----------
        x_train : ndarray
            (n_sample, n_dim)
        y_train : ndarray
            (n_sample, n_classes)
        dropout_layer : ndarray
            Dropout layer for the first hidden layer

        Returns
        ------
        zs : list
            zs[l] = (n_sample, n_units)
        activations : list
            activations[l] = (n_sample, n_units)
        losses : ndarray
            losses[i] = loss for sample i
        """
        zs = []
        al = x_train
        activations = [al]
        # loop over layers
        for idx, (wl, bl) in enumerate(zip(self.weights, self.biases)):
            # (n_sample, n_units) . (n_units, n_units2) + (n_sample, n_units2)
            zl = np.dot(al, wl) + bl
            al = sigmoid(zl)
            if idx == 0 and dropout_layer is not None:
                al *= dropout_layer
            zs.append(zl)
            activations.append(al)
        losses = self.get_losses(y_train, al)
        return zs, activations, losses

    def _backward_pass(self, x_train, y_train, zs, activations, dropout_layer):
        """Backpropagate the deltas and compute the gradient

        Parameters
        ----------
        x_train : ndarray
            (n_sample, n_dim)
        y_train : ndarray
            (n_sample, n_classes)
        zs : list
            zs[l] = (n_sample, n_units)
        activations : list
            activations[l] = (n_sample, n_units)

        Returns
        ------
        w_grads : list
            w_grads[l] = gradient for the weights, same shape as self.weights[l]
        b_grads = list
            b_grads[l] = gradient for the biases, same shape as self.biases[l]
        """
        n_sample, _ = x_train.shape

        # init deltas
        n_deltas = len(zs)
        deltas = [None] * n_deltas
        # (n, n_classes) o (n, n_classes)
        deltas[-1] = self.get_der_losses(y_train, activations[-1]) * der_sigmoid(zs[-1])

        # backpropagate the deltas
        for l in range(n_deltas - 2, -1, -1):
            # (n, n_units) = (n, n_units2) . (n_units2, n_units) + (n, n_units)
            deltas[l] = np.dot(deltas[l + 1], self.weights[l + 1].T) * der_sigmoid(zs[l])
            assert deltas[l].shape == zs[l].shape

        if dropout_layer is not None:
            deltas[0] *= dropout_layer

        # compute the gradient
        # w_grads[l] = (n, n_units, n_units2) = (n, n_units, :) o (n, :, n_units2)
        w_grads = [a[:, :, np.newaxis] * delta[:, np.newaxis, :]
                   for delta, a in zip(deltas, activations[:-1])]
        b_grads = deltas
        return w_grads, b_grads

    def _get_batch_estimates(self, x_train, y_train):
        """
        Compute the loss and the gradient on a mini batch

        Returns
        ------
        loss : float
        w_grads : list
            w_grads[l] = gradient for the weights, same shape as self.weights[l]
        b_grads = list
            b_grads[l] = gradient for the biases, same shape as self.biases[l]
        """
        if self.dropout > 0:
            # prepare dropout for the first hidden layer
            dropout_layer = nr.binomial(1, 1 - self.dropout, size=self.sizes[1]) / (1 - self.dropout)
        else:
            dropout_layer = None
        zs, activations, losses = self._forward_pass(x_train, y_train, dropout_layer=dropout_layer)
        w_grads, b_grads = self._backward_pass(x_train, y_train, zs, activations, dropout_layer)
        # average over the mini batch
        loss = np.mean(losses)
        w_grads = [np.mean(w_grad, axis=0) for w_grad in w_grads]
        b_grads = [np.mean(b_grad, axis=0) for b_grad in b_grads]

        # l2 reg
        if self.l2_reg > 0:
            loss += self.l2_reg * sum(np.sum(w ** 2) for w in self.weights)
            w_grads = [w_grad + 2 * self.l2_reg * w for w, w_grad in zip(self.weights, w_grads)]

        return loss, w_grads, b_grads

    def _update_model(self, w_grads, b_grads):
        """Update the network's weights and biases based on the gradient estimate"""
        for w, b, w_grad, b_grad in zip(self.weights, self.biases, w_grads, b_grads):
            w -= self.learning_rate * w_grad
            b -= self.learning_rate * b_grad

    def get_accuracy(self, x_val, y_val):
        _, activations, _ = self._forward_pass(x_val, y_val)
        y_true = np.argmax(y_val, axis=1)
        y_pred = np.argmax(activations[-1], axis=1)
        return np.sum(y_pred == y_true) / len(y_pred)

    def fit(self, x_train, y_train, batch_size=10, epochs=100, validation_data=None):
        """Fit the model using SGD"""
        assert len(x_train) == len(y_train)
        self.training_losses = []
        self.val_accs = []
        for epoch in range(epochs):
            perm = nr.permutation(len(x_train))
            for idx in range(0, len(x_train), batch_size):
                x_mini = x_train[perm[idx:idx+batch_size]]
                y_mini = y_train[perm[idx:idx+batch_size]]
                loss, w_grads, b_grads = self._get_batch_estimates(x_mini, y_mini)
                self.training_losses.append(loss)
                # save current model in case we need to restore it
                prev_weights = self.weights
                prev_biases = self.biases
                self._update_model(w_grads, b_grads)

            if validation_data:
                x_val, y_val = validation_data
                val_acc = self.get_accuracy(x_val, y_val)
                self.val_accs.append(val_acc)
                layer_l1_norms = [np.mean(np.abs(w)) for w in self.weights]
                logger.info("epoch=%s, val_acc=%s, layer_l1_norms=%s",
                            epoch, val_acc, layer_l1_norms)

                stop = (self.early_stopping and len(self.val_accs) >= 2 and
                        self.val_accs[-1] < self.val_accs[-2])
                if stop:
                    logger.info("Validation accuracy is decreasing, restoring previous model and stopping")
                    self.weights = prev_weights
                    self.biases = prev_biases
                    break

    def plot_statistics(self, exp_name='A'):
        """Plot learning statistics"""
        pd.Series(self.training_losses).plot()
        plt.xlabel('Mini batch')
        plt.ylabel('Training loss')
        plt.savefig(exp_name + '_training_loss.png')
        plt.close()

        if self.val_accs:
            pd.Series(self.val_accs).plot()
            plt.xlabel('Epoch')
            plt.ylabel('Validation Accuracy')
            plt.savefig(exp_name + '_val_acc.png')
            plt.close()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def der_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def squared_loss(y_true, y_pred):
    """
    Params
    ------
    y_true : ndarray
        (n_sample, n_classes)
    y_pred : ndarray
        (n_sample, n_classes)

    Returns
    ------
    ndarray
        (n_sample,)
    """
    return np.sum((y_true - y_pred) ** 2, axis=1) * 0.5


def der_squared_loss(y_true, y_pred):
    """
    Params
    ------
    y_true : ndarray
        (n_sample, n_classes)
    y_pred : ndarray
        (n_sample, n_classes)

    Returns
    ------
    ndarray
        (n_sample, n_classes)
    """
    return y_pred - y_true


def binary_cross_entropy(y_true, y_pred):
    """For one sample, this is the sum of binary cross entropies over each category

    Params
    ------
    y_true : ndarray
        (n_sample, n_classes)
    y_pred : ndarray
        (n_sample, n_classes)

    Returns
    ------
    ndarray
        (n_sample,)
    """
    n_sample, n_classes = y_true.shape
    assert np.sum(y_true == 1) == n_sample
    assert np.sum(y_true == 0) == n_sample * (n_classes - 1)
    return np.sum(
        - y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred), axis=1)


def der_binary_crossentropy(y_true, y_pred):
    """
    Params
    ------
    y_true : ndarray
        (n_sample, n_classes)
    y_pred : ndarray
        (n_sample, n_classes)

    Returns
    ------
    ndarray
        (n_sample, n_classes)
    """
    return (1 - y_true) / (1 - y_pred) - y_true / y_pred
