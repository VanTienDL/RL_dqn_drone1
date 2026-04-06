import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time

class DroneEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)

        self.A = np.array([0, 0, 1], dtype=np.float32)
        self.B = np.array([5, 5, 3], dtype=np.float32)

        self.step_size = 0.2
        self.max_steps = 200

        self.connect()

    def connect(self):
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

    def reset(self, seed=None, options=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        self.pos = self.A.copy()
        self.step_count = 0

        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.pos, self.B - self.pos]).astype(np.float32)

    def step(self, action):
        move = np.zeros(3)

        if action == 0: move[0] += self.step_size
        if action == 1: move[0] -= self.step_size
        if action == 2: move[1] += self.step_size
        if action == 3: move[1] -= self.step_size
        if action == 4: move[2] += self.step_size
        if action == 5: move[2] -= self.step_size

        self.pos += move
        self.step_count += 1

        dist = np.linalg.norm(self.pos - self.B)
        reward = -dist

        done = dist < 0.3 or self.step_count > self.max_steps

        return self._get_obs(), reward, done, False, {}

    def render(self):
        p.resetDebugVisualizerCamera(5, 45, -30, self.pos)
        time.sleep(1/60)