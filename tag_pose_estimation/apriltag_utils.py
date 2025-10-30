import numpy as np

def detections_to_corners_ids(
        detections: dict
) -> tuple[list, list]:
    if detections:
        # Corners should be Nx4x2
        corners = [np.expand_dims(d["lb-rb-rt-lt"].astype(np.float32),0) for d in detections]
        # Rearrange corners to lt-rt-rb-lb order for OpenCV
        corners = rearrange_corners(np.vstack(corners))
        # Make list again
        corners = [np.expand_dims(c.astype(np.float32), 0) for c in corners]

        # Get ids as Nx1 array of int32
        ids = np.array([[d['id']] for d in detections], dtype=np.int32)
    else:
        corners = []
        ids = None
    
    return corners, ids

def rearrange_corners(corners: np.ndarray) -> np.ndarray:
    """
    Rearrange corners from C1: [lb, rb, rt, lt] to C2: [lt, rt, rb, lb]. 
    Apriltag uses C1 convention when it returns detected corners, while 
    OpenCV Aruco functions use C2 convention.

    Parameters
    ----------
    corners : np.ndarray
        Array of shape Nx4x2 representing N detected markers with 4 corners
        in order [lb, rb, rt, lt].

    Returns
    -------
    np.ndarray
        Array of shape Nx4x2 representing N detected markers with 4 corners
        in order [lt, rt, rb, lb].
    """
    if corners.ndim != 3 or corners.shape[1:] != (4, 2):
        raise ValueError("Input corners must be of shape Nx4x2.")
    
    # Define the new order of indices
    new_order = [3, 2, 1, 0]
    
    # Rearrange the corners using advanced indexing
    rearranged_corners = corners[:, new_order, :]
    
    return rearranged_corners