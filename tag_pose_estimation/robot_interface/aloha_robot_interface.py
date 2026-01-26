import numpy as np
# Set to True to enable ALOHA robot interface
ALOHA = False
if ALOHA:
    from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
    from interbotix_common_modules.common_robot.robot import (
        create_interbotix_global_node,
        robot_startup,
        robot_shutdown,
    )
    from aloha.robot_utils import move_arm_to_ee_target_pose
    from aloha.constants import (
        CA_R_CA_EE,
        ROT_MAT_CA_EE,
    )
    from tag_pose_estimation.robot_interface.pose_transformations import (
        get_inline_pose_from_T,
        get_T_from_inline_pose,
    )

class AlohaRobotInterface:
    """
    Robot interface implementation for the ALOHA robot using the
    InterbotixManipulatorXS class from the interbotix_xs_modules package.
    """
    def __init__(self) -> None:
        # Start robot communication 
        self._node = create_interbotix_global_node('aloha')

        self._robot = InterbotixManipulatorXS(
            robot_model='vx300s',
            robot_name='follower_right',
            node=self._node,
            iterative_update_fk=True,
        )

        robot_startup(self._node)

    def get_arm_current_ee_pose(self) -> dict[str,np.ndarray]:
        """
        Get the current end-effector pose of the robot arm in 1x7 pose representation.
        Returns:
            dict: A dictionary containing:
                - "pose_inline": 1x7 numpy array representing the pose (x, y, z, qx, qy, qz, qw)
                - "pose_matrix": 4x4 numpy array representing the homogeneous transformation matrix
        """
        current_ee_pose = self._robot.arm.get_ee_pose()
        robot_ee_pose_inline = get_inline_pose_from_T(current_ee_pose)

        return {
            "pose_inline": robot_ee_pose_inline,
            "pose_matrix": current_ee_pose,
        }
    
    def move_arm_to_target_pose(self, target_pose: np.ndarray) -> None:
        """
        Move the robot arm to the specified target end-effector pose. The target
        pose should be in 1x7 pose representation.
        Parameters:
            target_pose: 1x7 numpy array representing the pose (x, y, z, qx, qy, qz, 
            qw)
        """
        commanded_ee_target_t = get_T_from_inline_pose(target_pose)
        move_arm_to_ee_target_pose(
                self._robot, commanded_ee_target_t, moving_time=2.0)
        
        return None

    def shutdown(self):
        robot_shutdown(self._node)

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
        T_ca_ee = np.eye(4)
        T_ca_ee[0:3, 0:3] = ROT_MAT_CA_EE
        T_ca_ee[0:3, 3] = CA_R_CA_EE

        return T_ca_ee