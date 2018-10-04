import numpy
numpy.set_printoptions(numpy.nan)
import numpy as np
import os
import sys

def sparse_inputs(img, depth, mask, Ah, Aw):
    """ Generate sparse inputs

    Args:
        img  : input image
        depth: ground truth depth map for the given image
        mask : pixels at which the ground truth depth is to be retained
        Ah: Downsampling factor along the height
        Aw: Downsampling factor along the width

    Returns:
        sparse inputs
    """
    H, W, _ = img.shape
    valid_depth_points = depth > 0
    S1 = np.zeros((H, W), dtype=np.float32)
    S2 = np.zeros((H, W), dtype=np.float32)

    # Make sure the sampling points have valid depth
    dist_tuple = [(i, j) for i in range(H) for j in range(W) if valid_depth_points[i, j]]
    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1 and depth[i, j] == 0:
                print(i, j)
                dist_transform = [np.sqrt(
                    np.square(i-vec[0])+
                    np.square(j-vec[1])) 
                    for vec in dist_tuple if vec != [i, j]
                ]
                closest_pixel = np.argmin(dist_transform)
                mask[i, j] = 0
                x, y = dist_tuple[closest_pixel]
                mask[x, y] = 1

    Sh = [i for i in range(H) for j in range(W) if mask[i, j]]
    Sw = [j for i in range(H) for j in range(W) if mask[i, j]]
    idx_to_depth = {}
    for i, (x, y) in enumerate(zip(Sh, Sw)):
        idx_to_depth[i] = x * W + y

    Sh, Sw = np.array(Sh), np.array(Sw)
    Hd, Wd = np.empty((H, W)), np.empty((H, W))
    Hd.T[:, ] = np.arange(H)
    Wd[:, ] = np.arange(W)
    Hd, Wd = Hd[..., None], Wd[..., None]
    Hd2 = np.square(Hd - Sh)
    Wd2 = np.square(Wd - Sw)
    dmap = np.sqrt(Hd2 + Wd2)
    dmap_arg = np.argmin(dmap, axis=-1)
    dmap_arg = dmap_arg.ravel()
    dmap_arg = np.array([idx_to_depth[i] for i in dmap_arg])
    S1 = depth.ravel()[dmap_arg].reshape(H, W)[None]
    S2 = np.sqrt(np.min(dmap, axis=-1))[None]
    return np.concatenate((S1, S2), axis=0)

def path_exists(path):
    """ Returns true if the path exists

    Args:
        path: path to a file/directory
    """
    return os.path.exists(path)

def generate_mask(Ah, Aw, H, W):
    """ Generates mask for the depth data
        Sparsity is: depth_values
    Args:
        Ah: Downsampling factor along h
        Aw: Downsampling factor along w
        H : Image height
        W : Image width

    Returns:
        binary mask of dimensions (H, W) where 1 equals 
        denotes actual ground truth data is retained
    """
    mask = np.zeros((H, W), dtype=np.bool)
    dh = np.rint(H * 1.0 / Ah).astype(np.int32)
    dw = np.rint(W * 1.0 / Aw).astype(np.int32)
    depth_values = dh * dw
    mask[0:None:Ah, 0:None:Aw] = 1
    return mask
