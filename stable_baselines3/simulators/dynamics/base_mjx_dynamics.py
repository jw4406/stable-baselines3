import jax
from jax import numpy as jp
import mujoco
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.io import html, mjcf, model
from mujoco import mjx
from mujoco import viewer
from .base_dynamics import BaseDynamics


class BaseMJXDynamics(BaseDynamics, PipelineEnv):

  def __init__(self, reset_noise_scale=1e-2, **kwargs):
    xml_path = "resources/mujoco_menagerie/unitree_go2/scene_mjx.xml"
    self.mj_model = mujoco.MjModel.from_xml_path(str(xml_path))

    # mjx opt settings
    self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    sys = mjcf.load_model(self.mj_model)

    physics_steps_per_control_step = 5
    kwargs['n_frames'] = kwargs.get('n_frames', physics_steps_per_control_step)
    kwargs['backend'] = 'mjx'

    super().__init__(sys, **kwargs)

    self._reset_noise_scale = reset_noise_scale

  def reset(self, rng: jp.ndarray) -> State:
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.qpos0 + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
    qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

    data = self.pipeline_init(qpos, qvel)

    obs = self._get_obs(data, jp.zeros(self.sys.nu))
    reward = 0
    done = 0
    metrics = {}
    return State(data, obs, reward, done, metrics)

  def integrate_forward(self, state: State, control: jp.ndarray, **kwargs) -> State:
    """Runs one timestep of the environment's dynamics."""
    data0 = state.pipeline_state
    data = self.pipeline_step(data0, action)

    obs = self._get_obs(data, action)
    reward = 0
    done = 0

    return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

  def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
    return jp.concatenate([data.qpos, data.qvel])
