import os
import time

import gymnax
import jax
import numpy as np

from MetaLearnCuriosity.utils import RandomAgentTransition as Transition
from MetaLearnCuriosity.wrappers import (
    BraxGymnaxWrapper,
    ClipAction,
    DelayedReward,
    LogWrapper,
    MiniGridGymnax,
    VecEnv,
)

environments = [
    "MiniGrid-BlockedUnlockPickUp",
    "MiniGrid-Empty-16x16",
    "MiniGrid-EmptyRandom-16x16",
    "MiniGrid-FourRooms",
    "MiniGrid-MemoryS16",
    "MiniGrid-Unlock",
    "ant",
    "halfcheetah",
    "hopper",
    "humanoid",
    "humanoidstandup",
    "inverted_pendulum",
    "inverted_double_pendulum",
    "pusher",
    "reacher",
    "walker2d",
    "Asterix-MinAtar",
    "Breakout-MinAtar",
    "Freeway-MinAtar",
    "SpaceInvaders-MinAtar",
]


def random_rollout(rng, env, env_params):
    total_episodes = 30

    @jax.jit
    def step(runner_state, tmp):

        env_state, last_obs, rng = runner_state

        # SELECT ACTION
        rng, act_rng = jax.random.split(rng)
        action = env.action_space(env_params).sample(act_rng)

        # STEP ENV
        rng, rng_step = jax.random.split(rng)
        obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
        transition = Transition(done, action, reward, info)
        runner_state = (env_state, obsv, rng)
        return runner_state, transition

    runner_state = (env_state, obsv, rng)
    epi_num = 0
    epi_rets = np.zeros(total_episodes)
    while epi_num < total_episodes:
        runner_state, trans = step(runner_state, None)
        epi_num += trans.done
        epi_rets[epi_num - 1] = trans.info["returned_episode_returns"]

    return epi_rets


for env_name in environments[:6]:
    rng = jax.random.PRNGKey(123)
    env = MiniGridGymnax(env_name)
    env_params = env._env_params
    env = LogWrapper(env)
    rng, reset_rng = jax.random.split(rng)
    obsv, env_state = env.reset(reset_rng, env_params)

    t = time.time()
    epi_rets = random_rollout(rng, env, env_params)
    elapsed_time = time.time() - t
    print(elapsed_time)
    print(f"{env_name}: {epi_rets.mean()}\t {epi_rets.shape}\n")

    # Create the directory if it doesn't exist
    output_dir = os.path.expanduser(
        "~/MetaLearnCuriosity/MetaLearnCuriosity/experiments/random_agents/"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save the numpy array
    file_path = os.path.join(output_dir, f"{env_name}_epi_rets.npy")
    np.save(file_path, epi_rets)
print("Metrics saved successfully")

for env_name in environments[6:16]:
    rng = jax.random.PRNGKey(123)
    env, env_params = BraxGymnaxWrapper(env_name), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = DelayedReward(env, 10)
    # env_params = env._env_params
    rng, reset_rng = jax.random.split(rng)
    obsv, env_state = env.reset(reset_rng, env_params)

    t = time.time()
    epi_rets = random_rollout(rng, env, env_params)
    elapsed_time = time.time() - t
    print(elapsed_time)
    print(f"{env_name}: {epi_rets.mean()}\t {epi_rets.shape}\n")

    # Create the directory if it doesn't exist
    output_dir = os.path.expanduser(
        "~/MetaLearnCuriosity/MetaLearnCuriosity/experiments/random_agents/"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save the numpy array
    file_path = os.path.join(output_dir, f"{env_name}_epi_rets.npy")
    np.save(file_path, epi_rets)
print("Metrics saved successfully")

for env_name in environments[16:]:
    rng = jax.random.PRNGKey(123)
    env, env_params = gymnax.make(env_name)
    env = LogWrapper(env)
    rng, reset_rng = jax.random.split(rng)
    obsv, env_state = env.reset(reset_rng, env_params)

    t = time.time()
    epi_rets = random_rollout(rng, env, env_params)
    elapsed_time = time.time() - t
    print(elapsed_time)
    print(f"{env_name}: {epi_rets.mean()}\t {epi_rets.shape}\n")

    # Create the directory if it doesn't exist
    output_dir = os.path.expanduser(
        "~/MetaLearnCuriosity/MetaLearnCuriosity/experiments/random_agents/"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save the numpy array
    file_path = os.path.join(output_dir, f"{env_name}_epi_rets.npy")
    np.save(file_path, epi_rets)
print("Metrics saved successfully")
