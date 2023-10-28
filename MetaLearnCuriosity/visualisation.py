import gymnax
import jax
import jax.numpy as jnp
from gymnax.visualize import Visualizer

from MetaLearnCuriosity.agents.BYOL_explore_toy import (
    BYOLActorCritic,
    OnlineEncoder,
    byol_make_train,
)
from MetaLearnCuriosity.agents.ppo import PPOActorCritic, ppo_make_train
from MetaLearnCuriosity.checkpoints import Restore
from MetaLearnCuriosity.wrappers import FlattenObservationWrapper


def train_best_seed(rng, make_train, config):
    """
    Train an RL agent for the best seed using a given training function.

    Parameters:
    - rng (jax.random.PRNGKey): Random number generator key.
    - make_train (function): A function that creates a training process for the RL agent.
    - config (dict): Configuration parameters for the training process.

    Returns:
    - output: The result of the training process.
    """

    train_jit = jax.jit(make_train(config))
    output = train_jit(rng)
    return output


def find_best_seed_params(path, agent_type, make_train):
    """
    Find the best seed parameters for an RL agent from a saved checkpoint.

    Parameters:
    - path (str): Path to the checkpoint.
    - agent_type (str): The type of RL agent ("BYOL" or "PPO").
    - make_train (function): A function that creates a training process for the RL agent.

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
        if agent_type == "BYOL":
            rng = output["runner_state"][-4][best_seed]
            out = train_best_seed(rng, make_train, output["config"])
            runner_state = out["runner_state"]
            online_params = runner_state[1].params
        else:
            rng = output["runner_state"][-1][best_seed]
            out = train_best_seed(rng, make_train, output["config"])
            runner_state = out["runner_state"]
            online_params = None
        agent_params = runner_state[0].params
    else:
        if agent_type == "BYOL":
            online_params = runner_state[1].params
        else:
            online_params = None
        agent_params = runner_state[0].params

    return agent_params, online_params


def visualise_gymnax(env, path, agent_type, make_train):
    """
    Visualize the performance of an RL agent in a Gym environment and create an animation.

    Parameters:
    - env: The Gym environment to visualize.
    - path (str): Path to the checkpoint.
    - agent_type (str): The type of RL agent ("BYOL" or "PPO").
    """

    rng = jax.random.PRNGKey(0)
    state_seq, reward_seq = [], []
    rng, rng_reset = jax.random.split(rng)

    agent_params, online_params = find_best_seed_params(path, agent_type, make_train)
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
