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

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  '''
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  '''

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    loss_i = 0.0
    count_margin_gt_zero = 0
    dW_i = np.zeros(W.shape)
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss_i += margin
        count_margin_gt_zero += 1
        dW_i[:, j] = X[i]
    
    dW_i[:, y[i]] = -count_margin_gt_zero * X[i]
    loss += loss_i
    dW += dW_i

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW = dW / num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW = dW + 2 * reg * W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss1 = 0.0
  dW1 = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  scores1 = X.dot(W)        
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores1_correct = scores1[np.arange(num_train), y]   # 1 by N
  scores1_correct = np.reshape(scores1_correct, (num_train, -1))  # N by 1
  margins = scores1 - scores1_correct + 1    # N by C
  # print("margins: ", margins[0:10], sep='\n')
  margins = np.maximum(0,margins)
  margins[np.arange(num_train), y] = 0
  scores2 = margins.copy()
  # print("margins: ", margins[0:10], sep='\n')
  loss1 += np.sum(margins) / num_train
  loss1 += 0.5 * reg * np.sum(W * W)

  
  # compute the gradient
  margins[margins > 0] = 1
  row_sum1 = np.sum(margins, axis=1)                  # 1 by N
  
  margins[np.arange(num_train), y] = -row_sum1
  dW1 = np.dot(X.T, margins)/num_train + reg * W     # D by C
  


  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = np.dot(X, W)
  scores -= scores[np.arange(num_train), y].reshape(num_train, 1)
  ones_to_add = np.ones([num_train, num_classes])
  ones_to_add[np.arange(num_train), y] = 0
  scores = scores + ones_to_add
  # print(scores)
  scores_idx = scores > 0
  print("idx equals: ", (scores_idx== (margins>0) ).all(), sep='\n' )
  # print(scores_idx.shape, scores_idx[0:20].astype(np.float), sep='\n')
  scores = np.maximum(0, scores)

  # print("scores: ", scores[0:10], sep='\n')
  print("scores equals: ", (scores2==scores).all(), sep='\n' )
  # print("scores: ", scores[0:20], sep='\n')
  # the "+1" is not necessary for the ground truth class, subtracted as follow.
  loss = np.sum(scores) 
  loss /= num_train
  loss += reg * np.sum(W * W)
  
  # pass
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
  
  row_sum = np.sum(scores_idx, axis=1)                  # 1 by N
  scores_idx = scores_idx.astype(np.int)
  scores_idx[np.arange(num_train), y] = -row_sum
  print("row_sum equals: ", (row_sum==row_sum1).all(), sep='\n')
  print("margins equals idx : ", (margins==scores_idx).all(), sep='\n')
  print("margins: ", margins[0:10], sep='\n')
  print("idx: ", scores_idx[0:10], sep='\n')
  dW = np.dot(X.T, scores_idx)/num_train + reg * W     # D by C
  # pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return loss, dW
