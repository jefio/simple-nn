# Vanilla Feedforward Neural Network

## Feedforward Neural Network Implementation

The `NeuralNetwork` class in `neural_network.py` implements the following features:
- variable number of hidden layers
- variable nodes per layer
- logistic sigmoid activation for hidden layers
- logistic sigmoid activation for the output layer
- backpropagation and two loss functions: the squared loss and the binary cross entropy

The class also implements the following regularization methods:
- l2-norm regularization of the weights
- dropout (only for the first hidden layer)
- early stopping if the validation accuracy decreases

## Iris Dataset

### Dependencies

Tested on Python 3.6 and depends on NumPy, SciPy, Pandas, Matplotlib, and Scikit-learn.

### How to run

The following command will fit the iris dataset with a default network and default parameters,
prints the final test accuracy, and create the files `A_training_loss.png` (evolution of the
training loss) and `A_val_acc.png` (evolution of the validation accuracy):

python learn_iris.py

To get help:

python learn_iris.py --help

### Results & Comments

A network with one hidden layer of size 8 and no regularization already
achieves 100% accuracy on the test set (60% training / 20% validation / 20% test).

python learn_iris.py --sizes 8

Let's see the effect of regularization. First, we fit a bigger network on a smaller dataset
(20% training / 40% validation / 40% test). This results in overfitting, and a test accuracy of 40%.

python learn_iris.py --sizes 256 --epochs 500 --train-ratio 0.2

We can achieve a test accuracy of 73% if we regularize using the L2-norm:

python learn_iris.py --sizes 256 --train-ratio 0.2 --epochs 500 --l2-reg 0.005 --learning-rate 0.1

We can achieve a test accuracy of 95% if we regularize using dropout:

python learn_iris.py --sizes 256 --epochs 500 --train-ratio 0.2 --dropout 0.2
