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


def save_episode_return(path_to_extract, path_to_save):
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
    means = np.array(means)
    ci_highs = np.array(ci_highs)
    ci_lows = np.array(ci_lows)
    stds = np.array(stds)

    # Ensure the save directory exists
    save_path = os.path.join(path_to_save, "baseline", output["config"]["ENV_NAME"])
    os.makedirs(save_path, exist_ok=True)

    # Save the arrays
    means_file = os.path.join(save_path, "means_episode_return.npy")
    ci_highs_file = os.path.join(save_path, "ci_highs_episode_return.npy")
    ci_lows_file = os.path.join(save_path, "ci_lows_episode_return.npy")
    stds_file = os.path.join(save_path, "stds_episode_return.npy")

    np.save(means_file, means)
    np.save(ci_highs_file, ci_highs)
    np.save(ci_lows_file, ci_lows)
    np.save(stds_file, stds)

    # Print the sizes of the saved files in MB
    print(f"Size of means_episode_return.npy: {os.path.getsize(means_file) / (1024 * 1024):.7f} MB")
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


environments = [
    "MiniGrid-BlockedUnlockPickUp",
    "MiniGrid-Empty-16x16",
    "MiniGrid-EmptyRandom-16x16",
    "MiniGrid-FourRooms",
    "MiniGrid-MemoryS128",
    "MiniGrid-Unlock",
]
for env_name in environments:
    save_episode_return(
        f"/home/batsi/Documents/Masters/MetaLearnCuriosity/minigrid-ppo-baseline_{env_name}_flax-checkpoints_v0",
        "/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments",
    )
