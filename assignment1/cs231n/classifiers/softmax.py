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
  num_train = X.shape[0]
  num_class = W.shape[1]
  ploss_pscore = np.zeros([num_train, num_class])
  scores = X.dot(W)
  # print(scores.shape)
  for i in xrange(num_train):
    scores[i] -= np.max(scores[i])
    loss += - np.log( np.exp(scores[i][y[i]]) / np.sum(np.exp(scores[i])))
    # print(np.exp(scores[i]).shape)
    ploss_pscore[i] = np.exp(scores[i]) / np.sum(np.exp(scores[i]))
    ploss_pscore[i][y[i]] -= 1
    # print(ploss_pscore.shape)
  
  loss = loss/ num_train
  loss += 0.5*reg * np.sum(W * W)
  # print(loss)
  ploss_pscore /= num_train
  dW = np.dot(X.T, ploss_pscore) + reg*W
  
  # pass
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  
  scores = X.dot(W)
  ploss_pscore = np.zeros([num_train, num_class])

  exp_scores = np.exp(scores)
  loss = np.sum( -np.log(np.exp(scores[np.arange(num_train), y]) / np.sum(exp_scores, axis=1)))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  ploss_pscore = exp_scores / np.sum(exp_scores, axis=1).reshape(num_train, 1)
  ploss_pscore[np.arange(num_train), y] -= 1
  ploss_pscore /= num_train

  dW = np.dot(X.T, ploss_pscore) + reg * W
  
  # pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

