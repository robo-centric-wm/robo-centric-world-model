import gym
import numpy as np
import os
from metaworld.envs import (
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
    ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
)


class MetaWorld:
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0, use_mask=False, use_proprio=False):
        os.environ["MUJOCO_GL"] = "egl"
        task = f"{name}-v2-goal-observable"
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_cls(seed=seed)
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat

        self._camera = camera
        self.reward_range = [-np.inf, np.inf]
        self.use_proprio = use_proprio
        self.use_mask = use_mask

        self.joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6',
                            'r_close', 'l_close']
        self.joint_ranges = self.get_joint_range()

    def get_joint_range(self):
        model = self._env.sim.model
        joint_lows = []
        joint_highs = []
        for joint_name in self.joint_names:
            joint_id = model.joint_name2id(joint_name)
            joint_range = model.jnt_range[joint_id].copy()
            joint_lows.append(joint_range[0])
            joint_highs.append(joint_range[1])
        joint_lows = np.array(joint_lows)
        joint_highs = np.array(joint_highs)
        joint_ranges = [joint_lows, joint_highs]
        return joint_ranges

    def norm_joint_pose(self, pose):
        joint_lows, joint_highs = self.joint_ranges
        # norm to [-1,1]
        norm_pose = 2 * (pose - joint_lows) / ((joint_highs - joint_lows) + 1e-10) - 1
        fix_joint = np.where(joint_lows == joint_highs)
        norm_pose[fix_joint] = pose[fix_joint]
        return norm_pose

    def get_proprio_state(self):
        data = self._env.sim.data
        qpos = []
        qvel = []
        for joint_name in self.joint_names:
            qpos.append(data.get_joint_qpos(joint_name).copy())
            qvel.append(data.get_joint_qvel(joint_name).copy())
        qpos = np.array(qpos)
        qvel = np.array(qvel)

        return qpos, qvel

    def get_proprio_mask(self):
        seg = self._env.sim.render(*self._size, mode="offscreen", camera_name=self._camera, segmentation=True)
        seg = seg[..., 1]
        body_names = (
            'right_l1', 'right_torso_itb', 'right_l2', 'right_l3', 'right_l4', 'right_l5', 'right_arm_itb',
            'right_hand_camera', 'right_wrist', 'right_l6', 'right_hand', 'hand', 'rightclaw', 'rightpad',
            'leftclaw', 'leftpad', 'right_l4_2', 'right_l2_2', 'right_l1_2')
        body_ids = [self._env.model.body_name2id(name) for name in body_names]
        body_ids.extend([i + max(body_ids) for i in range(5)])
        mask = np.where(np.isin(seg, body_ids), 1, 0)
        return np.expand_dims(mask, axis=-1)

    @property
    def observation_space(self):
        spaces = {}

        spaces['state'] = gym.spaces.Box(low=self._env.observation_space.low, high=self._env.observation_space.high,
                                         dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        if self.use_proprio:
            spaces["proprio_state"] = gym.spaces.Box(low=-1, high=1, shape=(18,), dtype=np.float32)

        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_space
        return spec

    @property
    def first_proprio_mask(self):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        _ = self._env.reset()
        proprio_mask = self.get_proprio_mask()
        return proprio_mask

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        success = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, info = self._env.step(action)
            reward += rew or 0
            success += float(info["success"])
        success = min(success, 1.0)
        assert success in [0.0, 1.0]
        obs = {
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ),
            "state": state,
            "success": success,
            "is_first": False,
            "is_terminal": False,  # will be handled by per_episode function

        }
        if self.use_mask:
            robot_mask = self.get_proprio_mask()
            obs["robot_mask"] = robot_mask

        if self.use_proprio:
            proprio_qpos, proprio_qvel = self.get_proprio_state()
            proprio_qpos = self.norm_joint_pose(proprio_qpos)
            proprio_state = np.hstack((proprio_qpos, proprio_qvel))
            obs['proprio_state'] = proprio_state

        return obs, reward, done, info

    def reset(self):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        state = self._env.reset()
        obs = {
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ),
            "state": state,
            "success": False,
            "is_first": True,
            "is_terminal": False,
        }

        if self.use_mask:
            robot_mask = self.get_proprio_mask()
            obs["robot_mask"] = robot_mask

        if self.use_proprio:
            proprio_qpos, proprio_qvel = self.get_proprio_state()
            proprio_qpos = self.norm_joint_pose(proprio_qpos)
            proprio_state = np.hstack((proprio_qpos, proprio_qvel))

            obs['proprio_state'] = proprio_state

        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "offscreen") != "offscreen":
            raise ValueError("Only render mode 'offscreen' is supported.")
        return self._env.render(*self._size)


