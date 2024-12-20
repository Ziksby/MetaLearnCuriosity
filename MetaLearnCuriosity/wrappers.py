# Taken from here: https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/wrappers.py

from functools import partial
from typing import Optional, Tuple, Union

import chex
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import xminigrid
from brax import envs
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from flax import struct
from gymnax.environments import environment, spaces
from xminigrid.wrappers import GymAutoResetWrapper


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

    def observation_space(self, env_params):
        return self._env.observation_space(env_params)

    def action_space(self, env_params):
        return self._env.action_space(env_params)


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    sum_of_rewards: float
    timestep: int
    done_sum_of_rewards: float


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, float, dict]:
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            sum_of_rewards=state.sum_of_rewards + (reward),
            done_sum_of_rewards=state.done_sum_of_rewards + reward * (1 - done),
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        info["sum_of_rewards"] = state.sum_of_rewards
        info["done_sum_of_rewards"] = state.done_sum_of_rewards
        return obs, state, reward, done, info


class BraxGymnaxWrapper:
    def __init__(self, env_name, backend="positional"):
        env = envs.get_environment(env_name=env_name, backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.time_limit = 1000

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return (
            next_state.obs,
            next_state,
            next_state.reward,
            next_state.done > 0.5,
            {},
        )

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


class ClipAction(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)


class TransformObservation(GymnaxWrapper):
    def __init__(self, env, transform_obs):
        super().__init__(env)
        self.transform_obs = transform_obs

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self.transform_obs(obs), state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self.transform_obs(obs), state, reward, done, info


class TransformReward(GymnaxWrapper):
    def __init__(self, env, transform_reward):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return obs, state, self.transform_reward(reward), done, info


class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: environment.EnvState


class NormalizeVecObservation(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        state = NormalizeVecObsEnvState(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-8),
            state,
            reward,
            done,
            info,
        )


@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState


class NormalizeVecReward(GymnaxWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info


@struct.dataclass
class MinAtarDelayedRewardEnvState:
    delayed_reward: float
    env_state: environment.EnvState


class MinAtarDelayedReward(GymnaxWrapper):
    def __init__(self, env, step_interval):
        super().__init__(env)
        self.step_interval = step_interval

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        state = MinAtarDelayedRewardEnvState(delayed_reward=0.0, env_state=state)
        return obs, state

    def step(self, key, state, action, params=None):
        # Perform a step in the wrapped environment
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)

        # Update the accumulated reward
        new_delayed_reward = state.delayed_reward + reward

        # Get the current step count from the environment
        steps = env_state.env_state.time

        # Check if we should release the delayed reward (either at step_interval or end of episode)
        interval = steps % self.step_interval == 0
        return_full_reward = (
            done | interval
        )  # Reward is returned either at intervals or when the episode ends

        # Use jax.lax.cond to determine the reward to return
        returned_reward = jax.lax.cond(
            return_full_reward,
            lambda: new_delayed_reward,  # Return the accumulated reward
            lambda: 0.0,  # Otherwise, return no reward
        )

        # Reset the delayed reward if the episode ends or interval is reached
        next_delayed_reward = jax.lax.cond(
            return_full_reward,
            lambda: 0.0,  # Reset accumulated reward
            lambda: new_delayed_reward,  # Keep accumulating
        )

        # Update the state with the new delayed reward and environment state
        state = MinAtarDelayedRewardEnvState(
            delayed_reward=next_delayed_reward, env_state=env_state
        )

        return obs, state, returned_reward, done, info


@struct.dataclass
class DelayedRewardEnvState:
    delayed_reward: float
    env_state: environment.EnvState


class DelayedReward(GymnaxWrapper):
    def __init__(self, env, step_interval):
        super().__init__(env)
        self.step_interval = step_interval

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        state = DelayedRewardEnvState(delayed_reward=0.0, env_state=state)
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        new_delayed_reward = state.delayed_reward + reward
        steps = env_state.env_state.info["steps"]

        # Check if we should release the delayed reward (either at step_interval or end of episode)
        interval = steps % self.step_interval == 0
        return_full_reward = (
            done | interval
        )  # Reward is returned either at intervals or when the episode ends

        # Use jax.lax.cond to determine the reward to return
        returned_reward = jax.lax.cond(
            return_full_reward,
            lambda: new_delayed_reward,  # Return the accumulated reward
            lambda: 0.0,  # Otherwise, return no reward
        )

        # Reset the delayed reward if the episode ends or interval is reached
        next_delayed_reward = jax.lax.cond(
            return_full_reward,
            lambda: 0.0,  # Reset accumulated reward
            lambda: new_delayed_reward,  # Keep accumulating
        )

        # Update the state with the new delayed reward and environment state
        state = DelayedRewardEnvState(delayed_reward=next_delayed_reward, env_state=env_state)

        return obs, state, returned_reward, done, info


@struct.dataclass
class ProbabilisticRewardEnvState:
    env_state: environment.EnvState


class ProbabilisticReward(GymnaxWrapper):
    def __init__(self, env, zero_prob):
        super().__init__(env)
        self.zero_prob = zero_prob

    def reset(self, key, params=None):
        key, reset_key = jax.random.split(key)
        obs, state = self._env.reset(reset_key, params)
        state = ProbabilisticRewardEnvState(env_state=state)
        return obs, state

    def step(self, key, state, action, params=None):
        # Split the key for the environment step and for our random reward zeroing
        key, step_key, random_key = jax.random.split(key, 3)

        obs, env_state, reward, done, info = self._env.step(
            step_key, state.env_state, action, params
        )

        # Use the random_key for generating the random number
        random_num = jax.random.uniform(random_key)

        # Use jax.lax.cond to conditionally zero out the reward
        returned_reward = jax.lax.cond(
            self.zero_prob == 0.0,
            lambda: reward,
            lambda: jax.lax.cond(random_num < self.zero_prob, lambda: 0.0, lambda: reward),
        )

        state = ProbabilisticRewardEnvState(env_state=env_state)
        return obs, state, returned_reward, done, info


class MiniGridGymnax:
    def __init__(self, env_name):
        env, env_params = xminigrid.make(env_name)
        env = GymAutoResetWrapper(env)
        self._env = env
        self._env_params = env_params
        self.time_limit = env.time_limit(env_params)

    def reset(self, key, env_params):
        timestep = self._env.reset(env_params, key)
        return timestep.observation, timestep

    def step(self, key, state, action, env_params):
        timestep = self._env.step(env_params, state, action)
        return timestep.observation, timestep, timestep.reward, timestep.last(), {}

    def observation_space(self, env_params):
        return spaces.Box(
            low=jnp.array([0, 0]),
            high=jnp.array([14, 13]),
            shape=(self._env.observation_shape(env_params),),
        )

    def action_space(self, env_params):
        return spaces.Discrete(6)
