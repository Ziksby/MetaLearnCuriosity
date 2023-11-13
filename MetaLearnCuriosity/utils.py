from typing import Any, NamedTuple, Sequence

import gymnax
import jax
import jax.numpy as jnp

from MetaLearnCuriosity.wrappers import FlattenObservationWrapper, LogWrapper


# Get dummy data for Obs Norm
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_obs_gymnax_discrete(config, env, env_params):
    def random_rollout(rng, env_params=env_params):
        """Rollout a jitted gymnax episode with lax.scan."""
        # Reset the environment
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        def step(runner_state, tmp):

            env_state, last_obs, rng = runner_state

            # SELECT ACTION
            rng, _rng, act_rngs = jax.random.split(rng, 3)
            act_rngs = jax.random.split(act_rngs, 4)
            action = jax.vmap(env.action_space(env_params).sample)(act_rngs)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                rng_step, env_state, action, env_params
            )
            transition = Transition(done, action, reward, last_obs, info)
            runner_state = (env_state, obsv, rng)
            return runner_state, transition

        runner_state = (env_state, obsv, rng)

        runner_state, traj_batch = jax.lax.scan(step, runner_state, None, config["NUM_STEPS"])

        return traj_batch.obs

    return random_rollout
