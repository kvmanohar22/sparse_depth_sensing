import numpy as np

def sparse_inputs(img, depth, mask):
    _, H, W = img.shape
    valid_depth_points = depth > 0
    S1 = np.zeros((H, W), dtype=np.float32)
    S2 = np.zeros((H, W), dtype=np.float32)

    # Make sure the sampling points have valid depth
    for i in range(H):
        for j in range(W):
            curr_pos = np.array([i, j], dtype=np.float32)
            if mask[i, j] == 1 and depth[i, j] > 0:
                r_new = np.argmin(np.square(curr_pos - valid_depth_points * depth))
                mask[r_new[0], r_new[1]] = 1
                mask[i, j] = 0
