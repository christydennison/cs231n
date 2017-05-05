import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dimensions = W.shape[0]

  for n in xrange(num_train):
    data = X[n]
    scores = data.dot(W)
    scores -= np.amax(scores) # subtract off max for numerical stability
    exp_scores = np.exp(scores) # exp scores
    sum_exp_scores = np.sum(exp_scores) # sum scores
    norm_scores = exp_scores/sum_exp_scores # normalize scores
    ce_loss = np.log(norm_scores)*-1 # calc loss
    correct_class = y[n]
    loss += ce_loss[correct_class] # add loss to total
    # find gradient
    for c in xrange(num_classes):
        if c == correct_class:
            dW[:,correct_class] += data*(norm_scores[correct_class]-1)
        else:
            dW[:,c] += data*(norm_scores[c])

  dW /= num_train
  dW += reg*2.0*W

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg*np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  tiled_max_scores = np.tile(np.amax(scores, axis=1),(num_classes,1)).T # max projected for rows, broadcast didn't work
  scores -= tiled_max_scores # subtract max from each row
  exp_scores = np.exp(scores) # exp scores
  rows_sum = np.sum(exp_scores, axis=1) # sum along rows
  norm_scores = np.divide(exp_scores,np.tile(rows_sum,(num_classes,1)).T) # divide each row by its sum, broadcast didn't work
  ce_loss = np.log(norm_scores)*-1 # take log of normalized scores
  loss = np.sum(ce_loss[np.arange(num_train),y])/num_train + reg*np.sum(W*W) # select loss for correct classes with reg
  dscores = np.copy(norm_scores)
  dscores[range(num_train),y] -= 1
  dscores /= num_train
  dW = np.dot(X.T, dscores)
  dW += reg*2.0*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

