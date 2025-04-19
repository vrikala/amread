import numpy as np


def integrate_los(start_pt: np.ndarray, end_pt: np.ndarray, R: np.ndarray, Z: np.ndarray, emission: np.ndarray, num_samples=10000):
    '''
    params:
    start_pt: np.ndarray, shape = (3,)
    end_pt:   np.ndarray, shape = (3,)
    R:        np.ndarray, shape = (n, 4)
    Z:        np.ndarray, shape = (n, 4)
    emission: np.ndarray, shape = (n,)

    Given a grid consisting of rectangles, and emission for each rectangle, this function calculates the line-integrated emission along the specified path. Supports 3D lines (not just top down)

    returns: float
    return params:
    total_integral: float
    '''
    # 1. Parametrize the line
    x0, y0, z0 = start_pt
    x1, y1, z1 = end_pt
    t = np.linspace(0, 1, num_samples)
    x = x0 + t * (x1 - x0)
    y = y0 + t * (y1 - y0)
    z = z0 + t * (z1 - z0)

    # 2. Convert to cylindrical coordinates
    R_points = np.sqrt(x**2 + y**2)
    Z_points = z

    # 3. Calculate differential arc lengths
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    ds = np.sqrt(dx**2 + dy**2 + dz**2)  # Lengths between sample points

    # 4. Precompute bounding boxes for each (R,Z) zone
    Rmin = np.min(R, axis=1)
    Rmax = np.max(R, axis=1)
    Zmin = np.min(Z, axis=1)
    Zmax = np.max(Z, axis=1)

    # 5. For each segment, determine which zone it's in
    total_integral = 0.0
    for i in range(len(ds)):
        Ri = R_points[i]
        Zi = Z_points[i]
        # Iterate over each cell
        for j in range(len(emission)):
            if Rmin[j] <= Ri <= Rmax[j] and Zmin[j] <= Zi <= Zmax[j]:
                total_integral += emission[j] * ds[i]
                break  # Each point can only belong to one zone
    
    return total_integral



