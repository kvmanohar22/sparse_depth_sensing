import numpy as np
# import cupy as cp

import chainer
import chainer.functions as F
import chainer.links as L

from chainer.backends import cuda
from chainer.backends.cuda import to_cpu
from chainer.backends.cuda import to_gpu

from utils.options import options
from densenet import DenseNet_conv

class D3(chainer.Chain):

    def __init__(self, opts):
        super(D3, self).__init__()

        self.L = opts['L']
        self.k = opts['k']
    
        with self.init_scope():
            self.conv1 = L.Convolution2D(5, 64, ksize=3, stride=2, pad=1)

            # DenseNet Block 1
            self.d1_conv1 = L.Convolution2D(64, 64, ksize=3, stride=1, pad=1)
            self.dense1 = DenseNet_conv(in_c=64)
            self.d1_conv2 = L.Convolution2D(self.k, 64, ksize=3, stride=2, pad=1)

            # DenseNet Block 2
            self.d2_conv1 = L.Convolution2D(64, 64, ksize=3, stride=1, pad=1)
            self.dense2 = DenseNet_conv(in_c=64)
            self.d2_conv2 = L.Convolution2D(self.k, 64, ksize=3, stride=2, pad=1)

            # DenseNet Block 3
            self.d3_conv1 = L.Convolution2D(64, 64, ksize=3, stride=1, pad=1)
            self.dense3 = DenseNet_conv(in_c=64)
            self.d3_conv2 = L.Convolution2D(self.k, 64, ksize=3, stride=2, pad=1)

            # DenseNet Block 4
            self.d4_conv1 = L.Convolution2D(64, 64, ksize=3, stride=1, pad=1)
            self.dense4 = DenseNet_conv(in_c=64)
            self.d4_conv2 = L.Convolution2D(self.k, 64, ksize=3, stride=2, pad=(2, 1))

            # DenseNet Block 5
            self.d5_conv1 = L.Convolution2D(64, 64, ksize=3, stride=1, pad=1)
            self.dense5 = DenseNet_conv(in_c=64)
            self.d5_conv2 = L.Convolution2D(self.k, 64, ksize=3, stride=2, pad=(1, 1))

            # DenseNet Connecting Block 2
            self.d2c_conv1 = L.Convolution2D(64, 64, ksize=3, stride=1, pad=1)
            self.dense2c = DenseNet_conv(in_c=64)
            self.d2c_conv2 = L.Convolution2D(self.k, 64, ksize=3, stride=2, pad=1)

            # DenseNet Connecting Block 3
            self.d3c_conv1 = L.Convolution2D(64, 64, ksize=3, stride=1, pad=1)
            self.dense3c = DenseNet_conv(in_c=64)
            self.d3c_conv2 = L.Convolution2D(self.k, 64, ksize=3, stride=2, pad=1)

            # DenseNet Connecting Block 4
            self.d4c_conv1 = L.Convolution2D(64, 64, ksize=3, stride=1, pad=1)
            self.dense4c = DenseNet_conv(in_c=64)
            self.d4c_conv2 = L.Convolution2D(self.k, 64, ksize=3, stride=2, pad=(2, 1))

            # DenseNet Block upsample 1
            self.up1_conv1 = L.Deconvolution2D(64, 64, ksize=2, stride=2, pad=0)
            self.up_dense1 = DenseNet_conv(in_c=64)
            self.up1_conv2 = L.Deconvolution2D(self.k, 64, ksize=3, stride=1, pad=1)

            # DenseNet Block upsample 2
            self.up2_conv1 = L.Deconvolution2D(64*2, 64, ksize=2, stride=2)
            self.up_dense2 = DenseNet_conv(in_c=64)
            self.up2_conv2 = L.Deconvolution2D(self.k, 64, ksize=3, stride=1, pad=1)

            # DenseNet Block upsample 3
            self.up3_conv1 = L.Deconvolution2D(64, 64, ksize=2, stride=2)
            self.up_dense3 = DenseNet_conv(in_c=64)
            self.up3_conv2 = L.Deconvolution2D(self.k, 64, ksize=3, stride=1, pad=1)

            # DenseNet Block upsample 4
            self.up4_conv1 = L.Deconvolution2D(64, 64, ksize=2, stride=2)
            self.up_dense4 = DenseNet_conv(in_c=64)
            self.up4_conv2 = L.Deconvolution2D(self.k, 64, ksize=3, stride=1, pad=1)

            # Final DeConvolution
            self.up5_conv1 = L.Deconvolution2D(64, 1, ksize=2, stride=2)

    def __call__(self, img, sparse_inputs):

        # Downsampling layers
        h = F.relu(self.conv1(np.concatenate((img, sparse_inputs), axis=1)))

        # Dense1
        x = np.resize(sparse_inputs, (1, 2, 240, 320))
        h = F.relu(self.d1_conv1(h))
        h = self.dense1(h, x)
        h = F.relu(self.d1_conv2(h))

        # Skip Dense 2
        _, _, H, W = x.shape
        x120 = np.resize(x, (1, 2, H//2, W//2))
        skip_h2 = F.relu(self.d2c_conv1(h))
        skip_h2 = self.dense2c(skip_h2, x120)
        skip_h2 = F.relu(self.d2c_conv2(skip_h2))

        h = F.relu(self.d2_conv1(h))
        h = self.dense2(h, x120)
        h = F.relu(self.d2_conv2(h))

        # Skip Dense 3
        _, _, H, W = x120.shape
        x60 = np.resize(x120, (1, 2, H//2, W//2))
        skip_h3 = F.relu(self.d3c_conv1(h))
        skip_h3 = self.dense3c(skip_h3, x60)
        skip_h3 = F.relu(self.d3c_conv2(skip_h3))

        h = F.relu(self.d3_conv1(h))
        h = self.dense3(h, x60)
        h = F.relu(self.d3_conv2(h))

        # Skip Dense 4
        _, _, H, W = x60.shape
        x30 = np.resize(x60, (1, 2, H//2, W//2))
        skip_h4 = F.relu(self.d4c_conv1(h))
        skip_h4 = self.dense4c(skip_h4, x30)
        skip_h4 = F.relu(self.d4c_conv2(skip_h4))

        h = F.relu(self.d4_conv1(h))
        h = self.dense4(h, x30)
        h = F.relu(self.d4_conv2(h))

        # Dense 5
        _, _, H, W = x30.shape
        x15 = np.resize(x30, (1, 2, H//2, W//2))
        b, c, H, W = x15.shape
        x16 = np.empty((b, c, H+1, W), dtype=np.float32)
        x16[:, :, :-1, ...] = x15
        x16[:, :, -1, ...] = x15[:, :, -1]
        h = F.relu(self.d5_conv1(h))
        h = self.dense5(h, x16)
        h = F.relu(self.d5_conv2(h))

        # UpDense 1
        print(h.shape)
        h = F.relu(self.up1_conv1(h))
        print(h.shape)
        h = self.up_dense1(h, x16)
        h = F.relu(self.up1_conv2(h))

        h = F.concat((h, skip_h4), axis=1)

        # UpDense 2
        print(h.shape)
        h = F.relu(self.up2_conv1(h))
        print(h.shape)
        # h = self.up_dense2(h, new_x)
        # h = F.relu(self.up2_conv2(h))
        # print(h.shape)

        # # UpDense 3
        # h = F.relu(self.up3_conv1(h))
        # h = self.up_dense3(h, new_x)
        # h = F.relu(self.up3_conv2(h))

        # # UpDense 4
        # h = F.relu(self.up4_conv1(h))
        # h = self.up_dense4(h, new_x)
        # h = F.relu(self.up4_conv2(h)) 

        return h

if __name__ == '__main__':
    opts = options().parse(train_mode=False)
    d3 = D3(opts)
    dummy_in = np.random.randn(1, 3, 480, 640).astype(np.float32)
    sparse_inputs = np.random.randn(1, 2, 480, 640).astype(np.float32)
    residual = d3(dummy_in, sparse_inputs)
    # print(residual.shape)
