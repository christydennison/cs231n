import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  delta = 1
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dimensions = W.shape[0]

  loss = 0.0
  for i in xrange(num_train):
    data = X[i]
    scores = data.dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:            
        continue
      margin = scores[j] - correct_class_score + delta
      if margin > 0:
        loss += margin
        dW[:,j] += data
        dW[:,y[i]] -= data

  dW /= num_train
  dW += reg*2.0*W

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg*np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  delta = 1.0
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  scores_correct = scores[np.arange(num_train),y] # select correct points
  scores_correct_tiled = np.tile(scores_correct,(num_classes,1)).T # broadcasting didn't work here
  loss_ma = scores - scores_correct_tiled + delta # get loss
  loss_ma[np.arange(num_train),y] = 0.0 # zero out correct class points
  loss_ma_masked = np.ma.masked_less_equal(loss_ma, 0.0).filled(0.0) # zero out <0
  loss = np.sum(loss_ma_masked)/num_train + reg*np.sum(W*W) # calc total loss
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  correct_mask = np.zeros(scores.shape) # create mask of the correct scores
  correct_mask[np.arange(num_train),y] = 1.0
  loss_ones = np.ma.masked_greater(loss_ma_masked, 0.0).filled(1.0) # losses turned into 1s, count_nonzero didn't work
  count_incorrect = np.tile(np.sum(loss_ones, axis=1),(num_classes,1)).T # sum along rows of incorrects, then tile
  correct_scaled_by_incorrect = np.multiply(correct_mask, count_incorrect) * -1 # scale corrects by number of incorrects
  gradient_counts = loss_ones + correct_scaled_by_incorrect # corrects and incorrects in same matrix
  dW = X.T.dot(gradient_counts)/num_train + reg*2.0*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
