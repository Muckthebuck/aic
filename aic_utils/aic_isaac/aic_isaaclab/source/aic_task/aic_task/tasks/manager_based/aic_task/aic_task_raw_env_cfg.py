# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AIC Task environment config with raw 224×224 RGB images instead of ResNet18 features.

Registers as ``AIC-Task-Raw-v0``. Identical to ``AIC-Task-v0`` except:
  - Camera observations use ``mdp.image`` (raw pixels) instead of ``mdp.image_features``
  - Images are in a separate observation group (``images``) so they stay as [B,H,W,3] tensors
  - Proprioception stays in ``policy`` group as a flat concatenated vector

Usage from RL code::

    env = gym.make("AIC-Task-Raw-v0", cfg=cfg)
    obs = env.reset()
    # obs["policy"]  -> flat proprio tensor [B, proprio_dim]
    # obs["images"]  -> dict with center_rgb, left_rgb, right_rgb as [B,224,224,3]
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .aic_task_env_cfg import (
    AICTaskEnvCfg,
    AICTaskSceneCfg,
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    RewardsCfg,
    TerminationsCfg,
)


@configclass
class RawImageObservationsCfg:
    """Observation config that keeps images as raw [B, 224, 224, 3] tensors.

    Two observation groups:
      - ``policy``: proprioceptive state (flat, concatenated) — uses absolute joint
        positions/velocities for easier mapping to NexusPrime's 68D format.
      - ``images``: raw RGB from 3 cameras (NOT concatenated, dict of tensors)

    Policy obs layout (75D total, concatenated):
      - joint_pos [6]: absolute joint positions (radians)
      - joint_vel [6]: absolute joint velocities (rad/s)
      - eef_pose [7]: TCP pose [x, y, z, qw, qx, qy, qz]
      - pose_command [7]: commanded TCP pose [x, y, z, qw, qx, qy, qz]
      - body_forces [42]: 7 bodies × 6D wrench (pre-scaled by 0.1)
      - actions [6]: last action (delta_pos[3] + delta_ori[3])
      - body_vel [1]: placeholder (reserved for future use)
    """

    @configclass
    class ProprioCfg(ObsGroup):
        """Proprioceptive observations — flat vector with absolute joint values."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        eef_pose = ObsTerm(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link")},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "ee_pose"}
        )
        body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "base_link",
                        "shoulder_link",
                        "upper_arm_link",
                        "forearm_link",
                        "wrist_1_link",
                        "wrist_2_link",
                        "wrist_3_link",
                    ],
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ImagesCfg(ObsGroup):
        """Raw 224×224 RGB images from 3 cameras — NOT concatenated.

        Each term produces [B, 224, 224, 3] float32 in [0, 1].
        Access via obs["images"]["center_rgb"], etc.
        """

        center_rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("center_camera"),
                "data_type": "rgb",
                "normalize": False,  # Raw uint8→float [0,255]; adapter applies ImageNet norm
            },
        )
        left_rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("left_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        right_rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("right_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False  # Keep as dict of [B, 224, 224, 3] tensors

    policy: ProprioCfg = ProprioCfg()
    images: ImagesCfg = ImagesCfg()


@configclass
class AICTaskRawEnvCfg(AICTaskEnvCfg):
    """AIC Task with raw 224×224 RGB images instead of ResNet18 features.

    Inherits everything from AICTaskEnvCfg, only overrides observations.
    """

    observations: RawImageObservationsCfg = RawImageObservationsCfg()
