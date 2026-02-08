import numpy as np
import sapien
import torch
import os
from copy import deepcopy

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor


# TODO (stao) (xuanlin): model it properly based on real2sim
@register_agent(asset_download_ids=["widowx250s"])
class WidowX250S(BaseAgent):
    uid = "widowx250s_custom"
    urdf_path = os.path.join(os.path.dirname(__file__), "wx250s.urdf")
    urdf_config = dict()

    GRIPPER_OPEN = 1.0
    GRIPPER_CLOSED = 0.0

    arm_joint_names = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    ]
    gripper_joint_names = ["left_finger", "right_finger"]
    ee_link_name = "ee_gripper_link"

    # PD controller parameters
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    keyframes = dict(
        home=Keyframe(
            # notice how we set the z position to be above 0, so the robot is not intersecting the ground
            pose=sapien.Pose(p=[-0.4, 0, 0]),
            qpos=np.zeros([len(arm_joint_names) + len(gripper_joint_names)]),
        )
    )

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm Controllers
        # -------------------------------------------------------------------------- #

        # PD Joint Position Control (Absolute position)
        arm_pd_joint_pos = PDJointPosControllerConfig(
            joint_names=self.arm_joint_names,
            lower=None,  # Will use joint limits from URDF
            upper=None,  # Will use joint limits from URDF
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
            use_delta=False,  # Absolute position control
        )

        # PD Joint Delta Position Control (Relative position)
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            joint_names=self.arm_joint_names,
            lower=-0.2,
            upper=0.2,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,  # Delta/incremental control
            normalize_action=False,
        )

        # PD EE Position Control (Absolute position in Cartesian space)
        arm_pd_ee_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-1.0,  # Workspace bounds (m)
            pos_upper=1.0,  # Workspace bounds (m)
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,  # Absolute EE position control
        )

        # PD EE Delta Position Control (Relative position in Cartesian space)
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=True,  # Delta EE position control
        )

        # PD EE Pose Control (Absolute pose: position + rotation)
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-1.0,  # Workspace bounds (m)
            pos_upper=1.0,  # Workspace bounds (m)
            rot_lower=-3.14,  # Rotation bounds (rad)
            rot_upper=3.14,  # Rotation bounds (rad)
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,  # Absolute EE pose control
            frame="root_translation:root_aligned_body_rotation",
        )

        # PD EE Delta Pose Control (Relative pose: position + rotation)
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,  # Position delta bounds (m)
            pos_upper=0.1,
            rot_lower=-0.15,  # Rotation delta bounds (rad)
            rot_upper=0.15,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=True,  # Delta EE pose control
            normalize_action=False,
            frame="root_translation:root_aligned_body_rotation",
        )

        # PD EE Target Delta Pose (Delta control based on target pose)
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # -------------------------------------------------------------------------- #
        # Gripper Controller
        # -------------------------------------------------------------------------- #
        # gripper_pd_joint_pos = PDJointPosControllerConfig(
        #     self.gripper_joint_names,
        #     lower=0.0,
        #     upper=1,  # Adjust based on actual gripper range
        #     stiffness=self.gripper_stiffness,
        #     damping=self.gripper_damping,
        #     force_limit=self.gripper_force_limit,
        # )

        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=0.0,
            upper=1.0,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=False,
            mimic={
                "left_finger": {
                    "joint": "right_finger",  # Left finger follows right finger
                    "multiplier": 1.0,  # Same direction (symmetric gripper)
                    "offset": 0.0,  # No offset
                }
            },
        )

        # -------------------------------------------------------------------------- #
        # Combined Controller Configurations
        # -------------------------------------------------------------------------- #
        controller_configs = dict(
            # Joint space control
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            # Cartesian space control - Position only
            pd_ee_pos=dict(arm=arm_pd_ee_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            # Cartesian space control - Position + Rotation
            pd_ee_pose=dict(arm=arm_pd_ee_pose, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos
            ),
        )

        return deepcopy(controller_configs)

    @property
    def tcp_pose(self):
        return self.robot.links_map[self.ee_link_name].pose

    @property
    def root_pose(self):
        root_link = self.robot.get_links()[0]  # FIXME: Please check whether this is root link
        root_pose = root_link.pose
        return root_pose

    def _after_loading_articulation(self):
        self.finger1_link = self.robot.links_map["left_finger_link"]
        self.finger2_link = self.robot.links_map["right_finger_link"]

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)
