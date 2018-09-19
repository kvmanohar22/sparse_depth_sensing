import numpy as np

def sparse_inputs(img, depth, mask):
    """ Generate sparse inputs

    Args:
        img  : input image
        depth: ground truth depth map for the given image
        mask : pixels at which the ground truth depth is to be retained

    Returns:
        sparse inputs
    """
    _, H, W = img.shape
    valid_depth_points = depth > 0
    S1 = np.zeros((H, W), dtype=np.float32)
    S2 = np.zeros((H, W), dtype=np.float32)

    # Make sure the sampling points have valid depth
    dist_tuple = [(i, j) for i in range(H) for j in range(W) if valid_depth_points[i, j]]
    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1 and depth[i, j] > 0:
                dist_transform = [np.sqrt(
                    np.square(i-vec[0])+
                    np.square(j-vec[1])) 
                    for vec in dist_tuple if vec != [i, j]
                ]
                closest_pixel = np.argmin(dist_transform)
                mask[i, j] = 0
                mask[closest_pixel[0], closest_pixel[1]] = 1
    
    # Construct sparse inputs
    sparse_depth_points = [(i, j) for i in range(H) for j in range(W) if mask[i, j]]
    for i in range(H):
        for j in range(W):
            r_nearest_vec = [np.sqrt(
                np.square(i-vec[0])+np.square(j-vec[1])
                for vec in sparse_depth_points
            )]
            r_nearest = np.argmin(r_nearest_vec)
            x, y = r_nearest
            S1[i, j] = depth[x, y]
            S2[i, j] = np.sqrt(np.sqrt(np.square(x-i)+np.square(y-j)))
    S1, S2 = S1[np.newaxis], S2[np.newaxis]
    return np.concatenate((S1, S2), axis=0)
