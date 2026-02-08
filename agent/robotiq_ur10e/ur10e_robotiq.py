from copy import deepcopy
import numpy as np
import sapien.core as sapien
import torch
import os

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
    PDEEPoseControllerConfig,
    PassiveControllerConfig,
)
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common


@register_agent()
class UR10eRobotiq(BaseAgent):
    """
    Combined UR10e robot with Robotiq ARG2f gripper using custom URDF.
    """

    uid = "ur10e_robotiq"
    urdf_path = os.path.join(os.path.dirname(__file__), "ur10e_robotiq.urdf")
    ee_link_name = "eef"

    GRIPPER_OPEN = 0.0
    GRIPPER_CLOSED = 1

    # Optional: tweak surface friction for gripper pads
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        home=Keyframe(
            qpos=np.concatenate(
                [
                    # UR10e 6-DOF in 'rest'
                    np.array([0, -1.5708, 1.5708, -1.5708, -1.5708, -1.5708]),
                    # Gripper both fingers open halfway
                    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ]
            ),
            pose=sapien.Pose(p=[-0.7, -0.1, 0.2]),
        )
    )

    @property
    def tcp_pose(self):
        return self.robot.links_map[self.ee_link_name].pose

    @property
    def root_pose(self):
        root_link = self.robot.get_links()[
            0
        ]  # FIXME: Please check whether this is root link
        root_pose = root_link.pose
        return root_pose

    @property
    def arm_joint_names(self):
        # first 6 qpos entries are the UR10e arm
        return [j.name for j in self.robot.active_joints[:6]]

    @property
    def gripper_joint_names(self):
        # names controlling the two outer knuckles
        return ["left_outer_knuckle_joint", "right_outer_knuckle_joint"]

    @property
    def _controller_configs(self):
        # Arm PD controllers
        arm_pd = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-6.28,
            upper=6.28,
            stiffness=1e4,
            damping=1e3,
            force_limit=200,
            normalize_action=False,
        )
        arm_pd_delta = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.5,
            upper=0.5,
            stiffness=1e4,
            damping=1e3,
            force_limit=200,
            normalize_action=False,
            use_delta=True,
        )

        # PD EE Pose Control (Absolute pose: position + rotation)
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            rot_lower=None,
            rot_upper=None,
            stiffness=1e4,
            damping=1e3,
            force_limit=200,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,  # Absolute control
            normalize_action=False,
            frame="root_translation:root_aligned_body_rotation",
        )

        # PD EE Delta Pose Control (Relative pose: position + rotation)
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.15,
            rot_upper=0.15,
            stiffness=1e4,
            damping=1e3,
            force_limit=200,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=True,
            normalize_action=False,
        )

        # Gripper PD controllers (mimic outer knuckles)
        passive = PassiveControllerConfig(
            joint_names=[
                "left_inner_knuckle_joint",
                "right_inner_knuckle_joint",
                "left_inner_finger_joint",
                "right_inner_finger_joint",
            ],
            damping=0,
            friction=2.0,
        )
        grip_active = PDJointPosMimicControllerConfig(
            joint_names=self.gripper_joint_names,
            lower=0,
            upper=1,
            stiffness=2e3,
            damping=400,
            force_limit=100,
            normalize_action=False,
        )

        return deepcopy(
            dict(
                pd_joint_pos=dict(
                    arm=arm_pd,
                    gripper_active=grip_active,
                    gripper_passive=passive,
                ),
                pd_joint_delta_pos=dict(
                    arm=arm_pd_delta,
                    gripper_active=grip_active,  # grip_active_delta,
                    gripper_passive=passive,
                ),
                pd_ee_pose=dict(
                    arm=arm_pd_ee_pose,
                    gripper_active=grip_active,
                    gripper_passive=passive,
                ),
                pd_ee_delta_pose=dict(
                    arm=arm_pd_ee_delta_pose,
                    gripper_active=grip_active,
                    gripper_passive=passive,
                ),
            )
        )

    def _after_loading_articulation(self):
        # setup drives for finger pads like XArm6Robotiq example
        def setup_finger(inner_name, pad_name):
            inner = self.robot.active_joints_map[inner_name]
            pad = self.robot.active_joints_map[pad_name].get_child_link()
            # transform poses from debug script
            p_f = [-1.6e-08, 3.76e-02, 4.30e-02]
            p_p = [1.36e-09, -1.79e-02, 6.52e-03]
            drv = self.scene.create_drive(
                inner.get_child_link(), sapien.Pose(p_f), pad, sapien.Pose(p_p)
            )
            drv.set_limit_x(0, 0)
            drv.set_limit_y(0, 0)
            drv.set_limit_z(0, 0)

        setup_finger("left_inner_knuckle_joint", "left_inner_finger_joint")
        setup_finger("right_inner_knuckle_joint", "right_inner_finger_joint")

        # disable collisions among gripper parts
        gripper_links = [
            "robotiq_arg2f_base_link",
            "left_outer_knuckle",
            "left_inner_knuckle",
            "right_outer_knuckle",
            "right_inner_knuckle",
            "left_inner_finger",
            "right_inner_finger",
            "left_inner_finger_pad",
            "right_inner_finger_pad",
        ]
        for ln in gripper_links:
            link = self.robot.links_map[ln]
            link.set_collision_group_bit(group=2, bit_idx=31, bit=1)

    def is_grasping(self, obj, min_force=0.5, max_angle=85):
        # reuse XArm6Robotiq logic for contact check
        f1 = self.scene.get_pairwise_contact_forces(
            self.robot.links_map["left_inner_finger_pad"], obj
        )
        f2 = self.scene.get_pairwise_contact_forces(
            self.robot.links_map["right_inner_finger_pad"], obj
        )
        lforce = torch.linalg.norm(f1, axis=1)
        rforce = torch.linalg.norm(f2, axis=1)
        ld = self.robot.links_map[
            "left_inner_finger_pad"
        ].pose.to_transformation_matrix()[..., :3, 1]
        rd = self.robot.links_map[
            "right_inner_finger_pad"
        ].pose.to_transformation_matrix()[..., :3, 1]
        lang = common.compute_angle_between(ld, f1)
        rang = common.compute_angle_between(rd, f2)
        return torch.logical_and(
            lforce >= min_force, torch.rad2deg(lang) <= max_angle
        ) & torch.logical_and(rforce >= min_force, torch.rad2deg(rang) <= max_angle)

    def is_static(self, threshold: float = 0.2):
        # Consider only arm joints (first 6 qvel) for static check
        qvel = self.robot.get_qvel()[..., :6]
        # Max abs velocity per env
        maxv = torch.max(torch.abs(qvel), dim=-1).values
        return maxv <= threshold

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose for UR10e TCP."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    @property
    def gripper_qpos(self):
        """Get the current gripper joint positions."""
        joint = self.robot.active_joints_map["left_outer_knuckle_joint"]
        qpos = joint.qpos.cpu().numpy()
        # lower, upper = 0.0, 0.81
        # return float(np.clip((qpos - lower) / (upper - lower), 0.0, 1.0))
        return float(np.clip(qpos, 0.0, 0.81))

    # FIXME: Please check whether this is correct
