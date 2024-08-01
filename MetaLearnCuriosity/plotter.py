import os
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import bootstrap

from MetaLearnCuriosity.checkpoints import Restore


def plot_csv_file(filenames, metric, agent_types, log_graph: bool = False):
    """
    Plots data from CSV files, typically from Weights and Biases, with optional logarithmic scaling.

    Parameters:
        filenames (list of str): List of CSV file paths to plot.
        metric (str): The name of the metric being plotted.
        agent_types (list of str): List of labels for different agents or experiments.
        log_graph (bool, optional): Whether to use logarithmic scaling for the y-axis. Default is False.

    Returns:
        None
    """
    for filename, agent_type in zip(filenames, agent_types):
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(filename)

        # Extract the first and second columns by index
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        # Convert the second column data to float and take the logarithm using NumPy
        y = y.astype(float)

        if log_graph:
            y = np.log(y)

        # Plot the data
        plt.plot(x, y, label=f"{agent_type}")
        plt.xlabel("Step")
        plt.ylabel(f"Log of {metric}")
        plt.legend()
        plt.grid(True)

    # Save the plot as an image file in the 'MLC_logs' directory
    plt.savefig(f"MLC_logs/{metric}.png")


def plot_distribution(agent_type, path):
    output = Restore(path)
    reward = []
    for i in range(len(output["metrics"]["returned_episode_returns"])):
        reward.append(output["metrics"]["returned_episode_returns"][i].mean(-1).reshape(-1)[-1])

    plt.hist(reward, bins=30, edgecolor="black", range=(0, 1))
    plt.xlabel("Episode Return")
    plt.ylabel("Frequency")
    plt.savefig(f"{path}/{agent_type}_histogram_{output['config']['NUM_SEEDS']}.png")


def plot_CI(names, labels, alphas):
    plt.figure(figsize=(8, 6))
    means = []
    ci_lows = []
    ci_highs = []
    for name, label, alpha in zip(names, labels, alphas):
        path = f"MLC_logs/flax_ckpt/Empty-misc/{name}_empty_30"
        output = Restore(path)
        metric = output["metrics"]["returned_episode_returns"]
        env_name = output["config"]["ENV_NAME"]
        # avg among the num of evns
        metric = jnp.mean(metric, axis=-1)

        # A 2d array of shape num_seeds, update step
        metric = metric.reshape(metric.shape[0], -1)

        # Transpose to make it (update steps, num_seeds)
        metric = metric.T

        timestep_values = (metric[-1],)
        means.append(jnp.mean(metric[-1]))
        ci = bootstrap(
            timestep_values,
            jnp.mean,
            confidence_level=0.95,
            method="percentile",
        )
        ci_lows.append(ci.confidence_interval.low)
        ci_highs.append(ci.confidence_interval.high)

    means = np.array(means)
    ci_highs = np.array(ci_highs)
    ci_lows = np.array(ci_lows)

    error_bar = np.array([means - ci_lows, ci_highs - means])
    plt.errorbar(labels, means, yerr=error_bar, fmt="o", capsize=5, label="Mean Episode Return")
    plt.xlabel("RL Agent")
    plt.ylabel("Mean Episode Return")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15))
    plt.grid()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{env_name}_mean_seeds_CI.png")


def plot_sample_std(names, labels, alphas):
    """
    This function plots the sample std during training for each algorithm
    """
    plt.figure(figsize=(8, 6))
    for name, label, alpha in zip(names, labels, alphas):
        sample_std = []
        path = f"MLC_logs/flax_ckpt/Empty-misc/{name}_empty_30"
        output = Restore(path)
        metric = output["metrics"]["returned_episode_returns"]
        env_name = output["config"]["ENV_NAME"]
        # avg among the num of evns
        metric = jnp.mean(metric, axis=-1)

        # A 2d array of shape num_seeds, update step
        metric = metric.reshape(metric.shape[0], -1)

        # Transpose to make it (update steps, num_seeds)
        metric = metric.T

        for i in range(len(metric)):
            sample_std.append(jnp.std(metric[i], ddof=1))

        sample_std = jnp.array(sample_std)
        plt.plot(sample_std, label=label, alpha=alpha)

    plt.xlabel("Update Step")
    plt.ylabel("The Sample Standard Deviation")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncols=len(names))
    plt.grid()
    plt.savefig(f"{env_name}_mean_seeds_std.png")


def save_episode_return(path_to_extract, path_to_save, type_agent):
    start_time = time.time()
    output = Restore(path_to_extract)
    metric = output["metrics"]["returned_episode_returns"]
    config = output["config"]

    # Average among the number of environments
    metric = jnp.mean(metric, axis=-1)

    # A 2D array of shape (num_seeds, update step)
    metric = metric.reshape(metric.shape[0], -1)

    # Transpose to make it (update steps, num_seeds)
    metric = metric.T

    # Initialize lists to store means, confidence intervals, and standard deviations
    means = []
    ci_lows = []
    ci_highs = []
    stds = []

    for timestep_values in metric:
        mean_value = jnp.mean(timestep_values)
        means.append(mean_value)  # Store the mean for the current timestep

        ci = bootstrap(
            (timestep_values,),
            jnp.mean,
            confidence_level=0.95,
            method="percentile",
        )

        ci_lows.append(ci.confidence_interval.low)
        ci_highs.append(ci.confidence_interval.high)

        std_value = jnp.std(timestep_values, ddof=1)  # Sample standard deviation
        stds.append(std_value)

    # Convert lists to numpy arrays
    metric = np.array(metric)
    means = np.array(means)
    ci_highs = np.array(ci_highs)
    ci_lows = np.array(ci_lows)
    stds = np.array(stds)

    # Ensure the save directory exists
    save_path = os.path.join(path_to_save, type_agent, output["config"]["ENV_NAME"])
    os.makedirs(save_path, exist_ok=True)

    # Save the arrays
    metric_file = os.path.join(save_path, "metric_seeds_episode_return.npy")
    means_file = os.path.join(save_path, "means_episode_return.npy")
    ci_highs_file = os.path.join(save_path, "ci_highs_episode_return.npy")
    ci_lows_file = os.path.join(save_path, "ci_lows_episode_return.npy")
    stds_file = os.path.join(save_path, "stds_episode_return.npy")

    np.save(means_file, means)
    np.save(metric_file, metric)
    np.save(ci_highs_file, ci_highs)
    np.save(ci_lows_file, ci_lows)
    np.save(stds_file, stds)

    # Print the sizes of the saved files in MB
    print(f"Size of means_episode_return.npy: {os.path.getsize(means_file) / (1024 * 1024):.7f} MB")
    print(
        f"Size of metric_seeds_episode_return.npy: {os.path.getsize(metric_file) / (1024 * 1024):.7f} MB"
    )

    print(
        f"Size of ci_highs_episode_return.npy: {os.path.getsize(ci_highs_file) / (1024 * 1024):.7f} MB"
    )
    print(
        f"Size of ci_lows_episode_return.npy: {os.path.getsize(ci_lows_file) / (1024 * 1024):.7f} MB"
    )
    print(f"Size of stds_episode_return.npy: {os.path.getsize(stds_file) / (1024 * 1024):.7f} MB")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to run the code: {elapsed_time:.2f} seconds in {config['ENV_NAME']}")


def normalize_curious_agent_returns(
    baseline_path, random_agent_path, curious_agent_path, save_path
):
    start_time = time.time()

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Load random agent data and calculate the mean
    random_agent_returns = np.load(random_agent_path)
    random_agent_mean = np.mean(random_agent_returns)
    print(f"Random Agent Mean: {random_agent_mean}")

    # Load baseline data and get the last element (episode return)
    baseline_returns = np.load(baseline_path)
    baseline_last_episode_return = baseline_returns[-1]
    print(f"Baseline Last Episode Return: {baseline_last_episode_return}")
    print(f"Baseline Last Episode Return SHape: {baseline_last_episode_return.shape}")

    # Load curious agent data and get the last element (episode return)
    curious_agent_returns = np.load(curious_agent_path)
    curious_agent_last_episode_return = curious_agent_returns[-1]
    print(f"Curious Agent Last Episode Return: {curious_agent_last_episode_return}")
    print(f"Curious Agent Last Episode Return Shape: {curious_agent_last_episode_return.shape}")

    # Normalize the curious agent returns between 0 and 1
    normalized_curious_agent_returns = (curious_agent_last_episode_return - random_agent_mean) / (
        baseline_last_episode_return - random_agent_mean
    )
    print(f"Normalized Curious Agent Returns: {normalized_curious_agent_returns}")

    # Construct the save path using the curious algorithm type
    os.makedirs(save_path, exist_ok=True)

    # Save the normalized curious agent returns
    normalized_curious_agent_file = os.path.join(save_path, "normalized_curious_agent_returns.npy")
    np.save(normalized_curious_agent_file, normalized_curious_agent_returns)

    # Print the size of the saved file in MB
    print(
        f"Size of normalized_curious_agent_returns.npy: {os.path.getsize(normalized_curious_agent_file) / (1024 * 1024):.2f} MB"
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to run the code: {elapsed_time:.2f} seconds")

    return normalize_curious_agent_returns


def get_normalise_cis_macro_env(paths):
    metrics = []
    for path in paths:
        metrics.append(np.load(path))
    metrics = np.array(metrics).flatten()
    ci = bootstrap(
        (metrics,),
        jnp.mean,
        confidence_level=0.95,
        method="percentile",
    )

    return metrics.mean(), ci.confidence_interval.low, ci.confidence_interval.high


# def plot_error_bars_normalised_envs(envs, curious_paths, save_path,curious_type):
#     for path in curious_paths:
#         normalised


def plot_error_bars_macro_envs(curious_paths, macro_env_type: str, curious_algo_types):
    means = []
    ci_highs = []
    ci_lows = []

    for paths in curious_paths:
        mean, ci_low, ci_high = get_normalise_cis_macro_env(paths)
        means.append(mean)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)
    means = np.array(means)
    ci_lows = np.array(ci_lows)
    ci_highs = np.array(ci_highs)
    error_bar = np.array([means - ci_lows, ci_highs - means])
    error_bar = np.array([means - ci_lows, ci_highs - means])

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        curious_algo_types,
        means,
        yerr=error_bar,
        fmt="o",
        capsize=5,
        capthick=2,
        elinewidth=1.5,
        label="Normalised Mean Episode Return",
        color="blue",
        markersize=8,
    )

    plt.xlabel("Curious Algorithm", fontsize=12, fontweight="bold", fontname="DejaVu Sans")
    plt.ylabel(
        "Normalised Mean Episode Return", fontsize=12, fontweight="bold", fontname="DejaVu Sans"
    )

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=45, ha="right", fontsize=10, fontname="DejaVu Sans")
    plt.yticks(fontsize=10, fontname="DejaVu Sans")

    plt.tight_layout()
    plt.savefig(
        f"/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments/images/{macro_env_type}_mean_seeds_CI_normalised.png",
        dpi=300,
    )
    plt.show()


# Example usage
# plot_error_bars_macro_envs(curious_paths, "SomeMacroEnvType", ["Algo1", "Algo2", "Algo3"])


environments = [
    "MiniGrid-BlockedUnlockPickUp",
    "MiniGrid-Empty-16x16",
    "MiniGrid-EmptyRandom-16x16",
    "MiniGrid-FourRooms",
    "MiniGrid-MemoryS128",
    # "MiniGrid-Unlock"
]
# environments = [
#     "Asterix-MinAtar",
#     "Breakout-MinAtar",
#     "Freeway-MinAtar",
#     "SpaceInvaders-MinAtar",
# ]
# for env_name in environments:
# save_episode_return(f"/home/batsi/Documents/Masters/MetaLearnCuriosity/minigrid-ppo-baseline_{env_name}_flax-checkpoints_v0",
#                     f"/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments",
#                     "baseline")
# normalize_curious_agent_returns(f"/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments/baseline/{env_name}/metric_seeds_episode_return.npy",
#                                 f"/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments/random_agents/{env_name}_epi_rets.npy",
#                                 f"/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments/rnd/{env_name}/metric_seeds_episode_return.npy",
#                                 f"/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments/rnd/{env_name}",
#                                 )
byol_paths = []
rnd_paths = []
for env_name in environments:
    byol_paths.append(
        f"/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments/byol/{env_name}/normalized_curious_agent_returns.npy"
    )
    rnd_paths.append(
        f"/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments/rnd/{env_name}/normalized_curious_agent_returns.npy"
    )

curious_paths = [byol_paths, rnd_paths]

plot_error_bars_macro_envs(curious_paths, "MiniGrid", ["BYOL-Explore", "RND"])
