import numpy as np

class RobotInterface:
    """
    Abstract base class for robot interfaces. This class defines the interface
    methods that must be implemented by any specific robot interface.
    """
    def __init__(self) -> None:
        raise NotImplementedError(
            "RobotInterface is an abstract class. "
            "Please use a specific robot interface implementation."
        )
    
    def get_arm_current_ee_pose(self) -> dict[str,np.ndarray]:
        """
        Get the current end-effector pose of the robot arm in 1x7 pose representation.
        Returns:
            dict: A dictionary containing:
                - "pose_inline": 1x7 numpy array representing the pose (x, y, z, qx, qy, qz, qw)
                - "pose_matrix": 4x4 numpy array representing the homogeneous transformation matrix
        """
        raise NotImplementedError(
            "get_arm_current_ee_pose() must be implemented by subclasses."
        )
    
    def move_arm_to_target_pose(self, target_pose: np.ndarray) -> None:
        """
        Move the robot arm to the specified target end-effector pose. The target
        pose should be in 1x7 pose representation.
        Parameters:
            target_pose: 1x7 numpy array representing the pose (x, y, z, qx, qy, qz, 
            qw)
        """
        raise NotImplementedError(
            "move_arm_to_target_pose() must be implemented by subclasses."
        )
    
    def shutdown(self):
        """
        Shutdown the robot interface and clean up resources.
        """
        raise NotImplementedError(
            "shutdown() must be implemented by subclasses."
        )
    
    def get_CA_frame_to_EE_frame_transform(self) -> np.ndarray:
        """
        Get the transformation matrix from the calibration adapter (CA) frame to
        the end-effector (EE) frame.

        The calibration adapter frame is the frame which is detected when a tag-
        board is attached to the end-effector. This transformation is fixed and
        known, but depends on the physical configuration of the calibration
        adapter.

        Returns:
            4x4 numpy array representing the transformation matrix
        """
        raise NotImplementedError(
            "get_CA_frame_to_EE_frame_transform() must be implemented by subclasses."
        )