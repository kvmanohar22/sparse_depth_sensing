from sparse_depth_sensing.utils.utils import generate_mask

import h5py
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt


def main():
    """ Tests the mask generation for a given downsampling factor
    """
    datafile = '/home/ankush/i18/sparse_depth_sensing/dataset/labeled_dataset.mat'
    idx = 123

    f = h5py.File(datafile)
    img = f['images'][idx]
    depth = f['depths'][idx]

    mask = generate_mask(24, 24, 480, 640)

    fig = plt.figure()
    plt.imshow(mask)
    plt.show()


if __name__ == '__main__':
    main()
