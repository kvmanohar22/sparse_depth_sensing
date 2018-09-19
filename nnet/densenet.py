import numpy as np
# import cupy as cp

import chainer
import chainer.functions as F
import chainer.links as L

from chainer.backends import cuda
from chainer.backends.cuda import to_cpu
from chainer.backends.cuda import to_gpu

class DenseNet_conv(chainer.Chain):
    """ Generates a single DenseNet block

    Args:
        in_c (int): Number of input feature maps
        DL   (int): Number of layers in DenseNet block
        k    (int): Number of output feature maps of DenseNet block

    Note: Currently the number of layers is fixed at 5
    """
    def __init__(self, in_c, DL=5, k=12):
        super(DenseNet_conv, self).__init__()
        self.L = DL
        self.k = k

        with self.init_scope():
            self.L1_conv1 = L.Convolution2D(in_c+2, self.k*4, ksize=1, stride=1)
            self.L1_conv2 = L.Convolution2D(self.k*4, self.k, ksize=3, stride=1, pad=1)

            self.L2_conv1 = L.Convolution2D(self.k+2, self.k*4, ksize=1, stride=1)
            self.L2_conv2 = L.Convolution2D(self.k*4, self.k, ksize=3, stride=1, pad=1)

            self.L3_conv1 = L.Convolution2D(self.k+2, self.k*4, ksize=1, stride=1)
            self.L3_conv2 = L.Convolution2D(self.k*4, self.k, ksize=3, stride=1, pad=1)

            self.L4_conv1 = L.Convolution2D(self.k+2, self.k*4, ksize=1, stride=1)
            self.L4_conv2 = L.Convolution2D(self.k*4, self.k, ksize=3, stride=1, pad=1)

            self.L5_conv1 = L.Convolution2D(self.k+2, self.k*4, ksize=1, stride=1)
            self.L5_conv2 = L.Convolution2D(self.k*4, self.k, ksize=3, stride=1, pad=1)

    def __call__(self, prev_out, sparse_inputs):
        h = F.concat((sparse_inputs, prev_out), axis=1)
        h = F.relu(self.L1_conv1(h))
        h = F.relu(self.L1_conv2(h))

        h = F.concat((sparse_inputs, h), axis=1)
        h = F.relu(self.L2_conv1(h))
        h = F.relu(self.L2_conv2(h))

        h = F.concat((sparse_inputs, h), axis=1)
        h = F.relu(self.L3_conv1(h))
        h = F.relu(self.L3_conv2(h))

        h = F.concat((sparse_inputs, h), axis=1)
        h = F.relu(self.L4_conv1(h))
        h = F.relu(self.L4_conv2(h))

        h = F.concat((sparse_inputs, h), axis=1)
        h = F.relu(self.L5_conv1(h))
        h = F.relu(self.L5_conv2(h))

        return h
