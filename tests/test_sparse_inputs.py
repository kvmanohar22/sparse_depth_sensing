from sparse_depth_sensing.utils.utils import sparse_inputs
from sparse_depth_sensing.utils.utils import generate_mask

import h5py
import pickle
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

def main():
    """ Test sparse inputs generation
    """
    print('Testing sparse input generation')

    idx = 123
    datafile = '/home/ankush/i18/sparse_depth_sensing/dataset/labeled_dataset.mat'

    f = h5py.File(datafile)
    img = f['images'][idx]
    depth = f['depths'][idx]

    img = np.transpose(img, (2, 1, 0))
    depth = np.transpose(depth, (1, 0))
    mask = generate_mask(48, 48, 480, 640)
    sparse_data = sparse_inputs(img, depth, mask, Ah=48, Aw=48)
    S1, S2 = np.split(sparse_data, 1)[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(151)
    plt.imshow(img)

    ax2 = fig.add_subplot(152)
    plt.imshow(depth)

    ax3 = fig.add_subplot(153)
    plt.imshow(mask)

    ax4 = fig.add_subplot(154)
    plt.imshow(S1)

    ax5 = fig.add_subplot(155)
    plt.imshow(S2)

    plt.show()

if __name__ == '__main__':
    main()
