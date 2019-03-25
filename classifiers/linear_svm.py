# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:20:22 2019

@author: Owner
"""

import numpy as np
from random import shuffle 

def svm_loss_vectorized(W, X, y, reg):
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
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #Find all scores
  #HERE: scores has dimension (num_data, num_classes)
  scores = np.dot(X, W)
  #Extract the scores of all y_i
  #Using two arrays indices to extract each location of scores.
  #one is arange to generate vector of size (num_data, 1)
  #second one is y, which is the labels vector.
  #Combine these vector will give the location (arange[i], y[i]) in the scores matrix
  correct_class_scores = scores[np.arange(num_train),y]
    
  #Find the margin between the scores using max(0, scores_j - scores_(y_i) + 1)
  margins = np.maximum(0, scores - np.transpose(np.array([correct_class_scores])) + 1)
  
  #Set the scores of the y_i to 0 since we want j != y_i
  margins[np.arange(num_train),y] = 0

  #Add all the margins together
  loss_i = np.sum(margins, axis=1)
  #Add all loss_i together and take the average
  loss = np.sum(loss_i) / num_train
  #Add regularization term
  loss += reg * np.sum(W * W)
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
  #Source http://cs231n.github.io/optimization-1/
  #Create a true/false matrix of the same size with margins.
  binary = margins
  #if the margins is more than zero, set to 1
  binary[margins > 0] = 1
  #if the margins is less than or equal zero, set to 0
  binary[margins <= 0] = 0
  #Find the sum of all binary values horizontally
  binary_sum = np.sum(binary, axis=1)
  #Set the value where j = y_i to -binary_sum.T  
  binary[np.arange(num_train), y] = -binary_sum.T
    
  #Multiply the binary to the data.
  dW = np.dot(X.T, binary)
  #Fina the average
  dW /= num_train

  #Add the Regularized term
  dW += 2*reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

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
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #Extract column where j = y_i
        dW[:,y[i]] -= X[i,:] 
        #Other columns.
        dW[:,j] += X[i,:] 
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW
