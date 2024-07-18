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
    "MiniGrid-DoorKey-16x16",
    "MiniGrid-Empty-16x16",
    "MiniGrid-EmptyRandom-16x16",
    "MiniGrid-FourRooms",
    "MiniGrid-LockedRoom",
    "MiniGrid-MemoryS128",
    "MiniGrid-Unlock",
    "MiniGrid-UnlockPickUp",
]


def random_rollout(rng, env, env_params):
    total_episodes = 10

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
    epi_rets = np.zeros(10)
    while epi_num < total_episodes:
        runner_state, trans = step(runner_state, None)
        epi_num += trans.done
        epi_rets[epi_num - 1] = trans.info["returned_episode_returns"]

    return epi_rets


for env_name in environments:
    rng = jax.random.PRNGKey(42)
    env = MiniGridGymnax(env_name)
    env_params = env._env_params
    env = LogWrapper(env)
    rng, reset_rng = jax.random.split(rng)
    obsv, env_state = env.reset(reset_rng, env_params)

    t = time.time()
    epi_rets = random_rollout(rng, env, env_params)
    elapsed_time = time.time() - t
    print(elapsed_time)
    print(epi_rets)

    # Create the directory if it doesn't exist
    output_dir = os.path.expanduser(
        "~/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments/random_agents/"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save the numpy array
    file_path = os.path.join(output_dir, f"{env_name}_epi_rets.npy")
    np.save(file_path, epi_rets)

print("Metrics saved successfully")
