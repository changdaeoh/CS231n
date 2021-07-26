from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_dim = X.shape[1]
    num_class = W.shape[1]
    for i in range(num_train):
        s = X[i].dot(W)
        true_class_idx = y[i]

        max_s = np.max(s)
        e_s = np.exp(s - max_s)  # substract max val for numerical stability
        e_t = np.sum(e_s)
        
        probs = e_s / e_t
        T_class_P = probs[true_class_idx]
        loss += (-1)*np.log(T_class_P)
        # -------------------
        dSM = -1/T_class_P

        # derivative of softmax (for true class)
        dS = (-1)*probs*T_class_P
        dS[true_class_idx] = T_class_P*(1-T_class_P)
        
        # chain rule -> dL/dSM, dSM/dS, dS/dW
        dW += dSM * (X[i].reshape(num_dim,1).dot(dS.reshape(1,num_class))) 

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train 
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    S = X.dot(W)                                    # (N, C)
 
    row_max = np.max(S, axis=1)                     # (N,)
    e_s = np.exp(S - row_max.reshape(num_train,1))  # (N, C)
    e_t = np.sum(e_s, axis = 1)                     # (N,)
    
    probs = e_s / e_t.reshape(num_train,1)          # (N, C)
    T_class_P = probs[list(range(num_train)), y]    # (N,)
    loss = np.sum((-1)*np.log(T_class_P))                

    loss /= num_train
    loss += reg * np.sum(W * W)
    #--------------------------- gradient
    dSM = -1/T_class_P                    # (N,)

    dS = (-1)*probs*(T_class_P.reshape(num_train,1)) # (N, C) * (N, 1) = (N, C)
    dS[list(range(num_train)),y] = T_class_P*(1-T_class_P)       
    dL_dS = dSM.reshape(num_train,1) * dS

    dW =  X.T.dot(dL_dS)

    dW /= num_train 
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW