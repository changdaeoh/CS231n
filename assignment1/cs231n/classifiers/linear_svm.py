from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
import copy

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    threshold = np.zeros((num_train, num_classes)) # (N x C)

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        fired_cnts = 0

        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                # loss 활성화 요소 개수 세기 (correct class에 대한 grad 계산을 위해)
                fired_cnts += 1
                # threshold 초과되는 위치 기억해놓기
                threshold[i, j] = 1
                loss += margin
        threshold[i, y[i]] = (-1)*fired_cnts

    # Right now the loss is a 'sum' over all training examples, but we want it
    # to be an 'average' instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss. (l2 reg)
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW = X.T.dot(threshold) # (D x N) * (N x C)
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # class score matrix
    S = X.dot(W)
    # correct class score들 구하기
    cc_score = S[list(range(num_train)), y] # (N,)
    # 브로드캐스팅 (N x C) - (N, ) => (N x C)
    margin = (S - cc_score.reshape(-1, 1)) + 1
    thresh_margin = np.maximum(0, margin)
    # 총 loss 구하고 N 빼주기(correct class col)
    loss = np.sum(thresh_margin) - num_train

    loss /= num_train
    loss += reg*np.sum(W*W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dS = copy.deepcopy(thresh_margin)
    # fired된 부분 1로 채우기
    dS[np.where(dS > 0)] = 1
    # fired 개수세고 1씩 빼주기
    fired_cnt = np.sum(dS, axis = -1) - 1
    # grad of correct class element 채워주기
    dS[list(range(num_train)), y] = (-1)*fired_cnt

    dW = X.T.dot(dS)

    dW /= num_train
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
