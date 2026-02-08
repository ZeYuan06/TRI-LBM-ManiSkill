from typing import Any, Dict, Union
import os
import numpy as np
import sapien
import torch
import trimesh

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig

from agent.robotiq_ur10e.ur10e_robotiq import UR10eRobotiq
from agent.widowx.widowx import WidowX250S


@register_env("PickBox-custom")
class PickBoxEnv(BaseEnv):
    """
    **Task Description:**
    A task where the objective is to grasp a box object and return it to the home position.
    The robot must pick up the box and bring it back to where the gripper started.

    **Randomizations:**
    - the box's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the box's z-axis rotation is randomized to a random angle

    **Success Conditions:**
    - the box is grasped by the robot
    - the robot's TCP is within `goal_thresh` (default 0.025m) euclidean distance of the home position
    - the robot is static (q velocity < 0.2)
    """

    SUPPORTED_ROBOTS = [
        UR10eRobotiq.uid,
        WidowX250S.uid,
    ]
    agent: Union[UR10eRobotiq, WidowX250S]
    uid = "PickBox-custom"
    goal_thresh = 0.05

    def __init__(
        self,
        *args,
        robot_uids=UR10eRobotiq.uid,
        robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=None,
        box_obj_path=os.path.join(
            os.path.dirname(__file__), "assets", "bleach_cleanser.obj"
        ),
        distractor_pool_paths=None,
        num_distractors=2,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.robot_uids = robot_uids
        # self.goal_thresh = 0.025
        self.box_obj_path = box_obj_path
        self.distractor_pool_paths = (
            distractor_pool_paths if distractor_pool_paths is not None else []
        )
        self.num_distractors = num_distractors

        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=100,
            control_freq=10,
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien.Pose()
        sensor_configs = [
            CameraConfig(
                "base_camera",
                pose,
                224,
                224,
                np.pi / 2,
                0.01,
                100,
                mount=self.cam_mount,
            ),
        ]
        if self.robot_uids == UR10eRobotiq.uid:
            mount_pose = sapien_utils.look_at(
                eye=[0.1, 0.0, 0.05], target=[0.0, 0.0, 0.35]
            )
            sensor_configs.append(
                CameraConfig(
                    uid="wrist_camera",
                    pose=mount_pose,
                    width=224,
                    height=224,
                    fov=np.pi / 2,
                    near=0.01,
                    far=100,
                    mount=self.agent.robot.links_map["wrist_3_link"],
                )
            )

        if self.robot_uids == WidowX250S.uid:
            mount_pose = sapien_utils.look_at(
                eye=[0.05, 0.0, 0.1], target=[0.35, 0.0, 0.0]
            )
            sensor_configs.append(
                CameraConfig(
                    uid="wrist_camera",
                    pose=mount_pose,
                    width=224,
                    height=224,
                    fov=np.pi / 2,
                    near=0.01,
                    far=100,
                    mount=self.agent.robot.links_map["gripper_link"],
                )
            )

        return sensor_configs

    @property
    def _default_human_render_camera_configs(self):
        render_pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return [
            CameraConfig("render_camera", render_pose, 256, 256, 1, 0.01, 100),
        ]

    def _load_agent(self, options: dict):
        super()._load_agent(options)

    # def _load_lighting(self, options: dict):
    #     for scene in self.scene.sub_scenes:
    #         scene.ambient_light = [
    #             np.random.uniform(0.2, 0.6),
    #             np.random.uniform(0.2, 0.6),
    #             np.random.uniform(0.2, 0.6),
    #         ]
    #         scene.add_directional_light(
    #             [1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096
    #         )
    #         scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cam_mount = self.scene.create_actor_builder().build_kinematic(
            "camera_mount"
        )  # camera mount

        # Load box object from assets
        box_obj_path = self.box_obj_path

        mesh = trimesh.load(box_obj_path)
        self.box_true_size = mesh.extents
        initial_z = self.box_true_size[2] / 2.0

        # Create box actor from mesh
        builder = self.scene.create_actor_builder()
        phys_mtl = self.scene.sub_scenes[0].create_physical_material(
            static_friction=4.0, dynamic_friction=4.0, restitution=0.0
        )
        builder.add_visual_from_file(box_obj_path)
        builder.add_convex_collision_from_file(
            box_obj_path, material=phys_mtl, density=100
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, initial_z])
        self.box = builder.build(name="box")

        self.no_grasp_objs = []
        self.obj_true_sizes = {}
        candidate_paths = []
        if len(self.distractor_pool_paths) > 0:
            # Filter out the target object from the pool
            # Normalize paths to ensure correct string comparison
            target_path = os.path.abspath(self.box_obj_path)
            candidate_paths = [
                p
                for p in self.distractor_pool_paths
                if os.path.abspath(p) != target_path
            ]

            max_pool_size = 20
            if len(candidate_paths) > max_pool_size:
                candidate_paths = np.random.choice(
                    candidate_paths, max_pool_size, replace=False
                )

        for i, obj_path in enumerate(candidate_paths):
            # Load mesh to get size
            try:
                mesh = trimesh.load(obj_path)
                obj_true_size = mesh.extents
            except Exception as e:
                print(f"Failed to load mesh for {obj_path}: {e}")
                continue

            initial_z = obj_true_size[2] / 2.0

            # Create builder
            obj_builder = self.scene.create_actor_builder()
            obj_builder.add_visual_from_file(obj_path)
            obj_builder.add_convex_collision_from_file(
                obj_path, material=phys_mtl, density=100
            )
            # Initialize far away so they don't collide initially
            obj_builder.initial_pose = sapien.Pose(p=[0, 0, -10.0 - i])

            # Build actor with unique name
            actor_name = f"distractor_{i}"
            actor = obj_builder.build(name=actor_name)

            self.no_grasp_objs.append(actor)
            self.obj_true_sizes[actor_name] = obj_true_size

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if self.robot_uids == UR10eRobotiq.uid:
                base_pose_np = sapien_utils.look_at(
                    eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1]
                )
            elif self.robot_uids == WidowX250S.uid:
                base_pose_np = sapien_utils.look_at(
                    eye=[0.45, 0, 0.4], target=[0.0, 0, 0.0]
                )
            else:
                raise NotImplementedError(f"Robot {self.robot_uids} not supported")
            base_pose = Pose.create(base_pose_np, device=self.device)
            pos_noise = torch.rand((b, 3), device=self.device) * 0.05 - 0.025
            rot_noise = randomization.random_quaternions(
                n=b, device=self.device, bounds=(-np.pi / 24, np.pi / 24)
            )
            noise_pose = Pose.create_from_pq(p=pos_noise, q=rot_noise)
            # final_cam_pose = base_pose * noise_pose
            final_cam_pose = base_pose
            self.cam_mount.set_pose(final_cam_pose)

            xyz = torch.zeros((b, 3))
            table_half_length = float(self.table_scene.table_length / 2.0 * 0.2)
            table_half_width = float(self.table_scene.table_width / 4.0 * 0.3)

            grasp_region = [
                [-table_half_length, -table_half_width],
                [table_half_length, table_half_width],
            ]
            distractor_scale = 3.0
            distractor_region = [
                [
                    -table_half_length * distractor_scale,
                    -table_half_width * distractor_scale,
                ],
                [
                    table_half_length * distractor_scale,
                    table_half_width * distractor_scale,
                ],
            ]

            # Initialize sampler with grasp_region for the target box
            sampler = randomization.UniformPlacementSampler(
                bounds=grasp_region, batch_size=b, device=self.device
            )

            # Place Target Box in grasp_region
            box_radius = (
                torch.linalg.norm(
                    torch.tensor(self.box_true_size[:2], dtype=torch.float32)
                )
                + 0.1
            )
            # Sample uniformly in grasp_region
            box_xy = sampler.sample(radius=box_radius, max_trials=100, verbose=False)
            xyz[:, :2] = box_xy
            xyz[:, 2] = self.box_true_size[2] / 2.0 + 0.01
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            # self.box.set_pose(Pose.create_from_pq(xyz, qs))
            self.box.set_pose(
                torch.Tensor([-0.1, 0, 0, 1, 0, 0, 0]).repeat(b, 1).to(self.device)
            )
            self.box.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.box.set_angular_velocity(torch.zeros((b, 3), device=self.device))

            # Place Distractors
            # far_away_pos = torch.zeros((b, 3), device=self.device)
            # for i, obj in enumerate(self.no_grasp_objs):
            #     far_away_pos[:, 2] = -11.0 - i
            #     obj.set_pose(Pose.create_from_pq(far_away_pos, qs))
            #     obj.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            #     obj.set_angular_velocity(torch.zeros((b, 3), device=self.device))

            new_bounds = torch.tensor(distractor_region, device=self.device)
            sampler._bounds = new_bounds
            sampler._ranges = new_bounds[1] - new_bounds[0]

            total_loaded = len(self.no_grasp_objs)
            num_active = min(self.num_distractors, total_loaded)

            if num_active > 0:
                # Randomly choose which actors to bring to the table
                active_indices = torch.randperm(total_loaded)[:num_active]

                for idx in active_indices:
                    obj = self.no_grasp_objs[idx]
                    obj_size = self.obj_true_sizes[obj.name]
                    obj_radius = torch.linalg.norm(torch.tensor(obj_size[:2])) + 0.1

                    obj_xy = sampler.sample(
                        radius=obj_radius, max_trials=100, verbose=False
                    )

                    is_zero = (torch.abs(obj_xy) < 1e-6).all(dim=1)
                    valid_mask = ~is_zero
                    if valid_mask.any():
                        target_z = obj_size[2] / 2.0 + 0.005
                        qs = randomization.random_quaternions(
                            b, lock_x=True, lock_y=True
                        )
                        final_p = torch.zeros((b, 3), device=self.device)  # All -10m
                        final_p[:, 2] = -10.0
                        final_p[valid_mask, :2] = obj_xy[valid_mask]
                        final_p[valid_mask, 2] = target_z

                        obj.set_pose(Pose.create_from_pq(final_p, qs))
                        obj.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                        obj.set_angular_velocity(
                            torch.zeros((b, 3), device=self.device)
                        )

        self.agent.robot.set_qpos(self.agent.keyframes["home"].qpos)
        self.agent.robot.set_pose(self.agent.keyframes["home"].pose)
        if self.robot_uids == UR10eRobotiq.uid:
            self.home_tcp_pose = torch.tensor(
                [-0.0086, 0.0741, 0.7239], device=self.device
            ).repeat(b, 1)
        elif self.robot_uids == WidowX250S.uid:
            self.home_tcp_pose = torch.tensor(
                [0.0583, 0, 0.3606], device=self.device
            ).repeat(b, 1)
        else:
            raise NotImplementedError("Unsupported robot for home tcp pose")

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )

        # if "state" in self.obs_mode:
        #     obs.update(
        #         obj_pose=self.box.pose.raw_pose,
        #         tcp_to_obj_pos=self.box.pose.p - self.agent.tcp_pose.p,
        #         tcp_to_home_pos = self.home_tcp_pose - self.agent.tcp_pose.p,
        #     )
        return obs

    def evaluate(self):
        is_grasped = self.agent.is_grasping(self.box)
        # is_robot_static = self.agent.is_static(0.2)
        tcp_to_home_dist = torch.linalg.norm(
            self.home_tcp_pose - self.agent.tcp_pose.p, axis=1
        )
        is_at_home = tcp_to_home_dist < self.goal_thresh

        return {
            "success": is_grasped & is_at_home,
            "is_at_home": is_at_home,
            # "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def visualize_pointcloud(
        self,
        obs: Dict,
        env_idx: int = 0,
        show_segmentation: bool = True,
        point_size: float = 1.0,
        max_points: int = 20000,
    ):
        """
        Visualize the pointcloud from observations.

        Args:
            obs: Observation dictionary containing 'pointcloud' key
            env_idx: Environment index to visualize (for batched environments)
            show_segmentation: Whether to color points by segmentation
            point_size: Size of the points in the visualization
            max_points: Maximum number of points to visualize for performance
        """
        import matplotlib.pyplot as plt

        if "pointcloud" not in obs:
            print("No pointcloud found in observations")
            return

        pc_data = obs["pointcloud"]

        # Extract point cloud data for the specified environment
        if isinstance(pc_data["xyzw"], torch.Tensor):
            points = pc_data["xyzw"][env_idx].cpu().numpy()
            colors = pc_data["rgb"][env_idx].cpu().numpy()
            segmentation = pc_data["segmentation"][env_idx].cpu().numpy()
        else:
            points = pc_data["xyzw"]
            colors = pc_data["rgb"]
            segmentation = pc_data["segmentation"]

        print(
            f"Original data shapes: points={points.shape}, colors={colors.shape}, seg={segmentation.shape}"
        )

        # Filter out invalid points (w != 0 typically indicates valid points)
        # valid_mask = points[:, 3] != 0
        valid_mask = np.ones_like(
            points[:, 3], dtype=bool
        )  # Assume all points are valid if no w channel
        valid_points = points[valid_mask][:, :3]  # Only use x, y, z
        valid_colors = colors[valid_mask]
        valid_segmentation = segmentation[valid_mask].flatten()

        # Downsample if too many points for visualization performance
        if len(valid_points) > max_points:
            print(
                f"Downsampling from {len(valid_points)} to {max_points} points for visualization"
            )
            indices = np.random.choice(len(valid_points), max_points, replace=False)
            valid_points = valid_points[indices]
            valid_colors = valid_colors[indices]
            valid_segmentation = valid_segmentation[indices]

        if len(valid_points) == 0:
            print("No valid points to visualize")
            return

        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        if show_segmentation and valid_segmentation is not None:
            # Color by segmentation
            unique_segments = np.unique(valid_segmentation)
            cmap = plt.cm.get_cmap("tab10")

            print(f"Found {len(unique_segments)} unique segments: {unique_segments}")

            for i, segment_id in enumerate(unique_segments):
                mask = valid_segmentation == segment_id
                points_in_segment = np.sum(mask)

                if points_in_segment > 0:
                    print(f"Segment {segment_id}: {points_in_segment} points")
                    segment_color = cmap(i % 10)
                    ax.scatter(
                        valid_points[mask, 0],
                        valid_points[mask, 1],
                        valid_points[mask, 2],
                        color=segment_color,
                        s=point_size,
                        label=f"Segment {segment_id} ({points_in_segment} pts)",
                        alpha=0.7,
                    )

            # Only show legend if we have a reasonable number of segments
            if len(unique_segments) <= 10:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            # Use RGB colors
            if len(valid_colors) > 0:
                if valid_colors.max() > 1.0:
                    valid_colors = valid_colors / 255.0  # Normalize if needed

                ax.scatter(
                    valid_points[:, 0],
                    valid_points[:, 1],
                    valid_points[:, 2],
                    c=valid_colors,
                    s=point_size,
                    alpha=0.7,
                )
            else:
                # Fallback to single color
                ax.scatter(
                    valid_points[:, 0],
                    valid_points[:, 1],
                    valid_points[:, 2],
                    color="blue",
                    s=point_size,
                    alpha=0.7,
                )

        # Set labels and title
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(
            f"Point Cloud Visualization (Env {env_idx}) - {len(valid_points)} points"
        )

        # Set equal aspect ratio
        if len(valid_points) > 0:
            ranges = np.array(
                [
                    valid_points[:, 0].max() - valid_points[:, 0].min(),
                    valid_points[:, 1].max() - valid_points[:, 1].min(),
                    valid_points[:, 2].max() - valid_points[:, 2].min(),
                ]
            )
            max_range = ranges.max() / 2.0

            mid_x = (valid_points[:, 0].max() + valid_points[:, 0].min()) * 0.5
            mid_y = (valid_points[:, 1].max() + valid_points[:, 1].min()) * 0.5
            mid_z = (valid_points[:, 2].max() + valid_points[:, 2].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plt.show()

        print(f"Visualized {len(valid_points)} valid points")
        if len(valid_points) > 0:
            print(f"Point cloud bounds:")
            print(
                f"  X: [{valid_points[:, 0].min():.3f}, {valid_points[:, 0].max():.3f}]"
            )
            print(
                f"  Y: [{valid_points[:, 1].min():.3f}, {valid_points[:, 1].max():.3f}]"
            )
            print(
                f"  Z: [{valid_points[:, 2].min():.3f}, {valid_points[:, 2].max():.3f}]"
            )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.box.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        tcp_to_home_dist = torch.linalg.norm(
            self.agent.tcp_pose.p - self.home_tcp_pose, axis=1
        )
        return_home_reward = 1 - torch.tanh(5 * tcp_to_home_dist)
        reward += return_home_reward * is_grasped

        qvel = self.agent.robot.get_qvel()
        if self.robot_uids == "panda":
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_at_home"] * is_grasped

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
