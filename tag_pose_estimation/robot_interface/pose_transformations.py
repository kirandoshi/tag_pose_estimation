import numpy as np
from scipy.spatial.transform import Rotation as R
from spatialmath import SO3, Quaternion

def get_pos_from_T(T):
    """
    Extract position (translation) from a 4x4 transformation matrix.

    Parameters:
        T: 4x4 numpy array representing the transformation matrix

    Returns:
        3-element numpy array representing the position (x, y, z)
    """
    return T[:3, 3]

def get_rot_from_T(T):
    """
    Extract rotation matrix from a 4x4 transformation matrix.

    Parameters:
        T: 4x4 numpy array representing the transformation matrix

    Returns:
        3x3 numpy array representing the rotation matrix
    """
    return T[:3, :3]

def get_inline_pose_from_T(T):
    """
    Converts 4x4 transformation matrix into 1x7 pose representation, where the 
    [ ee_pos (3), 
      ee_quat(4)]

    Parameters:
        T: 4x4 numpy array representing the transformation matrix

    Returns:
        1x7 numpy array representing the pose (x, y, z, qx, qy, qz, qw)
    """
    pos = get_pos_from_T(T)
    quat = R.from_matrix(get_rot_from_T(T)).as_quat()
    return np.concatenate((pos, quat))

def get_T_from_inline_pose(inline_pose):
    """
    Converts 1x7 pose representation into 4x4 transformation matrix.

    Parameters:
        inline_pose: 1x7 numpy array representing the pose (x, y, z, qx, qy, qz, 
        qw)
    
    Returns:
        4x4 numpy array representing the transformation matrix
    """
    pos = inline_pose[:3]
    quat = inline_pose[3:]
    rot = R.from_quat(quat)
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = rot.as_matrix()
    return T

def get_inline_pose_from_T_scalar_first(T):
    """
    Converts 4x4 transformation matrix into 1x7 pose representation, where the 
    [ ee_pos (3), 
      ee_quat(4)]
    The quaternion in the output pose has the scalar part first.

    Parameters:
        T: 4x4 numpy array representing the transformation matrix

    Returns:
        1x7 numpy array representing the pose (x, y, z, qw, qx, qy, qz)
    """
    pos = get_pos_from_T(T)
    quat = SO3(get_rot_from_T(T)).UnitQuaternion().vec
    return np.concatenate((pos, quat))

def get_T_from_inline_pose_scalar_first(inline_pose):
    """
    Converts 1x7 pose representation into 4x4 transformation matrix. The 
    quaternion in the inline_pose is expected to have the scalar part first.

    Parameters:
        inline_pose: 1x7 numpy array representing the pose (
            x, y, z, qw, qx, qy, qz)
    
    Returns:
        4x4 numpy array representing the transformation matrix
    """
    pos = inline_pose[:3]
    quat = inline_pose[3:]
    rot = Quaternion(quat).unit().SO3()
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = rot.R
    return T