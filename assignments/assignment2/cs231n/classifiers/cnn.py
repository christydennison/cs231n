from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        # N, C, H, W = x.shape
        # F, C, HH, WW = w.shape
        
        # conv layer (note that the padding and filter size aren't defined until the loss step)
        # H_p = W_p = 1 + (H+6-filter_size)/1 = 32
        # conv layer does not change the output H/W
        
        # pooling
        # stride = 2 ## 2x2 filter as specified above
        # HH = WW = 2 ## 2x2 filter as specified above
        # H_p = W_p = 1 + (H - HH) / stride = 1 + (32-2)/2 = 16
        # pooling layer effectively cuts the H/W in half
        
        # out = np.zeros((N, F, H_p, W_p))
        # input for layer after conv layer should be F*H_p*W_p = num_filters*H/2*W/2
        
        self.params['W1'] = np.random.normal(size=(num_filters, C, filter_size, filter_size), scale=weight_scale)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = np.random.normal(size=(num_filters*(H/2)*(W/2), hidden_dim), scale=weight_scale)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.normal(size=(hidden_dim, num_classes), scale=weight_scale)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        convOut, cacheConv = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        affRout, cacheAffR = affine_relu_forward(convOut, W2, b2)
        affout, cacheAff = affine_forward(affRout, W3, b3)
        scores = affout
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        lossSoftmax, dSoftmax = softmax_loss(affout, y)
        dxAff, dwAff, dbAff = affine_backward(dSoftmax, cacheAff)
        dxAffR, dwAffR, dbAffR = affine_relu_backward(dxAff, cacheAffR)
        dxConv, dwConv, dbConv = conv_relu_pool_backward(dxAffR, cacheConv)
        
        reg_loss = 0.5*(self.reg*np.sum(W1*W1) + self.reg*np.sum(W2*W2) + self.reg*np.sum(W3*W3))
        loss = lossSoftmax + reg_loss
        grads['W3'] = dwAff + self.reg * W3
        grads['W2'] = dwAffR + self.reg * W2
        grads['W1'] = dwConv + self.reg * W1
        grads['b3'] = dbAff
        grads['b2'] = dbAffR
        grads['b1'] = dbConv
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
