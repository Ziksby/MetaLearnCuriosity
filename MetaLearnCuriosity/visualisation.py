import gymnax
import jax
import jax.numpy as jnp
from gymnax.visualize import Visualizer

from MetaLearnCuriosity.agents.byol_explore_lite import BYOLActorCritic, OnlineEncoder
from MetaLearnCuriosity.agents.ppo import PPOActorCritic
from MetaLearnCuriosity.checkpoints import Restore
from MetaLearnCuriosity.wrappers import FlattenObservationWrapper


def extract_best_seed_params(output, best_seed, index):
    params = {}
    best_params = {}
    # Assuming all Dense layers have the same keys
    dense_keys = output["runner_state"][index]["params"]["params"].keys()

    for layer_key in dense_keys:
        params[layer_key] = {
            "kernel": output["runner_state"][index]["params"]["params"][layer_key]["kernel"][
                best_seed
            ],
            "bias": output["runner_state"][index]["params"]["params"][layer_key]["bias"][best_seed],
        }
    best_params["params"] = params
    return best_params


def find_best_seed_params(path, agent_type):
    """
    Find the best seed parameters for an RL agent from a saved checkpoint.

    Parameters:
    - path (str): Path to the checkpoint.
    - agent_type (str): The type of RL agent ("BYOL" or "PPO").

    Returns:
    - agent_params: Parameters for the agent network.
    - online_params: Parameters for the online encoder (or None if not using BYOL).
    """

    output = Restore(path)
    runner_state = output["runner_state"]

    if output["config"]["NUM_SEEDS"] > 1:
        performance_per_seed = []
        for i in range(output["config"]["NUM_SEEDS"]):
            current_mean = (
                output["metrics"]["returned_episode_returns"][i].mean(-1).reshape(-1)
            ).mean()
            performance_per_seed.append(current_mean)
        best_seed = jnp.argmax(jnp.array(performance_per_seed))
        agent_params = extract_best_seed_params(output, best_seed, 0)
        if agent_type == "BYOL":
            online_params = extract_best_seed_params(output, best_seed, 1)
        else:
            online_params = None
    else:
        if agent_type == "BYOL":
            online_params = runner_state[1]["params"]
        else:
            online_params = None
        agent_params = runner_state[0]["params"]

    return agent_params, online_params


def visualise_gymnax(env, path, agent_type):
    """
    Visualize the performance of an RL agent in a gymnax environment and create an animation.

    Parameters:
    - env: The gymnax environment to visualize.
    - path (str): Path to the checkpoint.
    - agent_type (str): The type of RL agent ("BYOL" or "PPO").
    """

    rng = jax.random.PRNGKey(0)
    state_seq, reward_seq = [], []
    rng, rng_reset = jax.random.split(rng)

    agent_params, online_params = find_best_seed_params(path, agent_type)
    env, env_params = gymnax.make(env)
    env = FlattenObservationWrapper(env)
    action_dim = env.action_space(env_params).n
    online = OnlineEncoder(64)

    if agent_type == "BYOL":
        agent = BYOLActorCritic(action_dim)
    else:
        agent = PPOActorCritic(action_dim)
    obs, state = env.reset(rng_reset, env_params)

    while True:
        state_seq.append(state)
        rng, rng_step = jax.random.split(rng, 2)
        if online_params is None:
            encoded_obs = obs
        else:
            encoded_obs = online.apply(online_params, obs)
        pi, _ = agent.apply(agent_params, encoded_obs)
        action = pi.sample(seed=rng)
        next_obs, next_state, reward, done, info = env.step(rng_step, state, action, env_params)
        reward_seq.append(reward)
        if done:
            break
        else:
            obs = next_obs
            state = next_state

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    vis = Visualizer(env, env_params, state_seq, cum_rewards)
    vis.animate(f"{path}/anim_{agent_type}.gif")
