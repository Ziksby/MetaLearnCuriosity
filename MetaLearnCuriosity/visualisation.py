import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from MetaLearnCuriosity.agents.byol_explore_lite import BYOLActorCritic, OnlineEncoder
from MetaLearnCuriosity.agents.ppo import PPOActorCritic
from MetaLearnCuriosity.checkpoints import Restore
from MetaLearnCuriosity.wrappers import FlattenObservationWrapper


def extract_seed_params(output, seed, index):
    params = {}
    best_params = {}
    # Assuming all Dense layers have the same keys
    dense_keys = output["runner_state"][index]["params"]["params"].keys()

    for layer_key in dense_keys:
        params[layer_key] = {
            "kernel": output["runner_state"][index]["params"]["params"][layer_key]["kernel"][seed],
            "bias": output["runner_state"][index]["params"]["params"][layer_key]["bias"][seed],
        }
    best_params["params"] = params
    return best_params


def find_best_seed_params(path, agent_type, n_best_seed=1):
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
    agent_params, online_params = [], []
    if n_best_seed > output["config"]["NUM_SEEDS"]:
        raise ValueError(
            f"Invalid value for n_seed. n_seed cannot be greater than NUM_SEEDS, which is {output['config']['NUM_SEEDS']}."
        )

    if output["config"]["NUM_SEEDS"] > 1:
        performance_per_seed = []
        for i in range(output["config"]["NUM_SEEDS"]):
            current_mean = (
                output["metrics"]["returned_episode_returns"][i].mean(-1).reshape(-1)
            ).mean()
            performance_per_seed.append(current_mean)

        top_n_seeds = jnp.argsort(jnp.array(performance_per_seed))[::-1][:n_best_seed]
        for seed in top_n_seeds:
            agent_param = extract_seed_params(output, seed, 0)
            if agent_type == "BYOL":
                online_param = extract_seed_params(output, seed, 1)
            else:
                online_param = None
            agent_params.append(agent_param)
            online_params.append(online_param)
    else:
        assert n_best_seed == 1, "n_seed must be equal to 1 when NUM_SEEDS is 1."
        if agent_type == "BYOL":
            online_params.append(runner_state[1]["params"])
        else:
            online_params.append(None)
        agent_params.append(runner_state[0]["params"])

    return agent_params, online_params


def visualise_gymnax(env, path, agent_type, n_best_seed=1):
    """
    Visualize the performance of an RL agent in a gymnax environment and create an animation.

    Parameters:
    - env: The gymnax environment to visualize.
    - path (str): Path to the checkpoint.
    - agent_type (str): The type of RL agent ("BYOL" or "PPO").
    """

    rng = jax.random.PRNGKey(0)
    rng, rng_reset = jax.random.split(rng)
    state_seqs = []
    agent_params, online_params = find_best_seed_params(path, agent_type, n_best_seed)
    env, env_params = gymnax.make(env)
    env = FlattenObservationWrapper(env)
    action_dim = env.action_space(env_params).n
    online = OnlineEncoder(64)
    seed_num = 0
    for agent_param, online_param in zip(agent_params, online_params):
        state_seq, reward_seq = [], []

        if agent_type == "BYOL":
            agent = BYOLActorCritic(action_dim)
        else:
            agent = PPOActorCritic(action_dim)
        obs, state = env.reset(rng_reset, env_params)

        while True:
            state_seq.append(state)
            rng, rng_step = jax.random.split(rng, 2)
            if agent_type == "BYOL":
                encoded_obs = online.apply(online_param, obs)
            else:
                encoded_obs = obs
            pi, _ = agent.apply(agent_param, encoded_obs)
            action = pi.sample(seed=rng)
            next_obs, next_state, reward, done, info = env.step(rng_step, state, action, env_params)
            reward_seq.append(reward)
            if done:
                print(next_state, done)
                break
            else:
                obs = next_obs
                state = next_state

        # cum_rewards = jnp.cumsum(jnp.array(reward_seq))
        # vis = Visualizer(env, env_params, state_seq, cum_rewards)
        # vis.animate(f"{path}/anim_{agent_type}_{seed_num}.gif")
        state_seqs.append(state_seq)
        seed_num += 1

    return state_seqs


def generate_heatmap(state_seqs, agent_type, path, grid_size=(16, 16)):
    # Create a grid to represent the positions
    heatmap = np.zeros(grid_size)

    # Extract the starting position from the first state in the first sequence
    start_pos = (state_seqs[0][0].pos[0], state_seqs[0][0].pos[1])

    # Iterate through each state sequence
    for state_seq in state_seqs:
        # Extract x and y positions from each EnvState in the sequence
        positions = [(state.pos[0], state.pos[1]) for state in state_seq]

        # Increment the corresponding positions in the heatmap
        for x, y in positions:
            heatmap[x, y] += 1

    # Normalize the heatmap values to be between 0 and the maximum visit frequency
    max_frequency = np.max(heatmap)

    # Extract the goal position from the last state in the last sequence
    goal_pos = (state_seqs[-1][-1].goal[0], state_seqs[-1][-1].goal[1])
    _, ax = plt.subplots(figsize=(8, 6))
    # Plot the normalized heatmap with a custom color map (viridis)
    im = ax.imshow(heatmap, cmap="plasma", interpolation="lanczos", vmin=0, vmax=max_frequency)

    # Mark the starting position with a blue square
    plt.scatter(start_pos[1], start_pos[0], color="blue", marker="s", s=100, label="Start Position")

    # Mark the goal position with a green triangle
    plt.scatter(goal_pos[1], goal_pos[0], color="green", marker="^", s=100, label="Goal Position")

    # plt.title('Agent\'s Visited States Heatmap')
    _ = ax.figure.colorbar(im, ax=ax)

    ax.set_xlabel("Y Position")
    ax.set_ylabel("X Position")
    ax.legend()

    plt.savefig(f"{path}/heatmap_{agent_type}_30.png")


path = "MLC_logs/flax_ckpt/Empty-misc_diff_config/fast_30"
s = visualise_gymnax("Empty-misc", path, "fast", n_best_seed=30)
generate_heatmap(s, "fast", path)
