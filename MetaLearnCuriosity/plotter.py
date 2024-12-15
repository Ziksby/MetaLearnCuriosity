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


def save_int_lambdas(path_to_extract, path_to_save, type_agent, env_name):
    start_time = time.time()
    output = Restore(path_to_extract)
    metric = output["int_lambdas"]
    # env_name = output["config"]["ENV_NAME"]
    print(f"\n Here's the shape:\n {metric.shape}")

    # # Average among the number of environments
    # metric = jnp.mean(metric, axis=-1)

    # A 2D array of shape (num_seeds, update step)
    metric = metric.reshape(metric.shape[0], -1)

    # Transpose to make it (update steps, num_seeds)
    metric = metric.T

    # Initialize lists to store means, confidence intervals, and standard deviations
    means = []
    # ci_lows = []
    # ci_highs = []
    # stds = []

    for timestep_values in metric:
        mean_value = jnp.mean(timestep_values)
        means.append(mean_value)  # Store the mean for the current timestep

    #     ci = bootstrap(
    #         (timestep_values,),
    #         jnp.mean,
    #         confidence_level=0.95,
    #         method="percentile",
    #     )

    #     ci_lows.append(ci.confidence_interval.low)
    #     ci_highs.append(ci.confidence_interval.high)

    #     std_value = jnp.std(timestep_values, ddof=1)  # Sample standard deviation
    #     stds.append(std_value)

    # Convert lists to numpy arrays
    metric = np.array(metric)
    means = np.array(means)
    # ci_highs = np.array(ci_highs)
    # ci_lows = np.array(ci_lows)
    # stds = np.array(stds)

    # Ensure the save directory exists
    save_path = os.path.join(path_to_save, type_agent, env_name)
    os.makedirs(save_path, exist_ok=True)

    # Save the arrays
    metric_file = os.path.join(save_path, "int_lambda_seeds_episode_return.npy")
    means_file = os.path.join(save_path, "means_int_lambda.npy")
    # ci_highs_file = os.path.join(save_path, "ci_highs_episode_return.npy")
    # ci_lows_file = os.path.join(save_path, "ci_lows_episode_return.npy")
    # stds_file = os.path.join(save_path, "stds_episode_return.npy")

    np.save(means_file, means)
    np.save(metric_file, metric)
    # np.save(ci_highs_file, ci_highs)
    # np.save(ci_lows_file, ci_lows)
    # np.save(stds_file, stds)

    # Print the sizes of the saved files in MB
    print(f"Size of means_episode_return.npy: {os.path.getsize(means_file) / (1024 * 1024):.7f} MB")
    print(
        f"Size of metric_seeds_episode_return.npy: {os.path.getsize(metric_file) / (1024 * 1024):.7f} MB"
    )

    # print(
    #     f"Size of ci_highs_episode_return.npy: {os.path.getsize(ci_highs_file) / (1024 * 1024):.7f} MB"
    # )
    # print(
    #     f"Size of ci_lows_episode_return.npy: {os.path.getsize(ci_lows_file) / (1024 * 1024):.7f} MB"
    # )
    # print(f"Size of stds_episode_return.npy: {os.path.getsize(stds_file) / (1024 * 1024):.7f} MB")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to run the code: {elapsed_time:.2f} seconds in {env_name}")


def save_episode_return(path_to_extract, path_to_save, type_agent, env_name):
    start_time = time.time()
    output = Restore(path_to_extract)
    metric = output["metrics"]["returned_episode_returns"]
    # env_name = output["config"]["ENV_NAME"]
    print(f"\n Here's the shape:\n {metric.shape}")

    # # Average among the number of environments
    # metric = jnp.mean(metric, axis=-1)

    # A 2D array of shape (num_seeds, update step)
    metric = metric.reshape(metric.shape[0], -1)

    # Transpose to make it (update steps, num_seeds)
    metric = metric.T

    # Initialize lists to store means, confidence intervals, and standard deviations
    means = []
    # ci_lows = []
    # ci_highs = []
    # stds = []

    for timestep_values in metric:
        mean_value = jnp.mean(timestep_values)
        means.append(mean_value)  # Store the mean for the current timestep

    #     ci = bootstrap(
    #         (timestep_values,),
    #         jnp.mean,
    #         confidence_level=0.95,
    #         method="percentile",
    #     )

    #     ci_lows.append(ci.confidence_interval.low)
    #     ci_highs.append(ci.confidence_interval.high)

    #     std_value = jnp.std(timestep_values, ddof=1)  # Sample standard deviation
    #     stds.append(std_value)

    # Convert lists to numpy arrays
    metric = np.array(metric)
    means = np.array(means)
    # ci_highs = np.array(ci_highs)
    # ci_lows = np.array(ci_lows)
    # stds = np.array(stds)

    # Ensure the save directory exists
    save_path = os.path.join(path_to_save, type_agent, env_name)
    os.makedirs(save_path, exist_ok=True)

    # Save the arrays
    metric_file = os.path.join(save_path, "metric_seeds_episode_return.npy")
    means_file = os.path.join(save_path, "means_episode_return.npy")
    # ci_highs_file = os.path.join(save_path, "ci_highs_episode_return.npy")
    # ci_lows_file = os.path.join(save_path, "ci_lows_episode_return.npy")
    # stds_file = os.path.join(save_path, "stds_episode_return.npy")

    np.save(means_file, means)
    np.save(metric_file, metric)
    # np.save(ci_highs_file, ci_highs)
    # np.save(ci_lows_file, ci_lows)
    # np.save(stds_file, stds)

    # Print the sizes of the saved files in MB
    print(f"Size of means_episode_return.npy: {os.path.getsize(means_file) / (1024 * 1024):.7f} MB")
    print(
        f"Size of metric_seeds_episode_return.npy: {os.path.getsize(metric_file) / (1024 * 1024):.7f} MB"
    )

    # print(
    #     f"Size of ci_highs_episode_return.npy: {os.path.getsize(ci_highs_file) / (1024 * 1024):.7f} MB"
    # )
    # print(
    #     f"Size of ci_lows_episode_return.npy: {os.path.getsize(ci_lows_file) / (1024 * 1024):.7f} MB"
    # )
    # print(f"Size of stds_episode_return.npy: {os.path.getsize(stds_file) / (1024 * 1024):.7f} MB")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to run the code: {elapsed_time:.2f} seconds in {env_name}")


def normalize_curious_agent_returns(
    baseline_path, random_agent_path, curious_agent_path, save_path
):
    start_time = time.time()

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Load random agent data and calculate the mean
    random_agent_returns = np.load(random_agent_path)
    print(f"Random Last Episode Return: {random_agent_returns}")

    random_agent_mean = np.mean(random_agent_returns)
    print(f"Random Agent Mean: {random_agent_mean}")

    # Load baseline data and get the last element (episode return)
    baseline_returns = np.load(baseline_path)
    baseline_last_episode_return = baseline_returns[-1]
    print(f"Baseline Last Episode Return: {baseline_last_episode_return}")
    print(f"Baseline Last Episode Return SHape: {baseline_last_episode_return.shape}")
    print(f"Baseline Mean: {baseline_last_episode_return.mean()}")

    # Load curious agent data and get the last element (episode return)
    curious_agent_returns = np.load(curious_agent_path)
    curious_agent_last_episode_return = curious_agent_returns[-1]
    print(f"Curious Agent Last Episode Return: {curious_agent_last_episode_return}")
    print(f"Curious Agent Last Episode Return Shape: {curious_agent_last_episode_return.shape}")

    # Normalize the curious agent returns between 0 and 1
    normalized_curious_agent_returns = (curious_agent_last_episode_return - random_agent_mean) / (
        baseline_last_episode_return.mean() - random_agent_mean
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


def get_normalise_cis_env(path):

    metrics = np.load(path)
    ci = bootstrap(
        (metrics,),
        jnp.mean,
        confidence_level=0.95,
        method="percentile",
    )
    return metrics.mean(), ci.confidence_interval.low, ci.confidence_interval.high


def plot_error_bars_env(curious_paths, env_name: str, curious_algo_types):
    means = []
    ci_highs = []
    ci_lows = []

    for path in curious_paths:
        mean, ci_low, ci_high = get_normalise_cis_env(path)
        means.append(mean)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)
    means = np.array(means)
    ci_lows = np.array(ci_lows)
    ci_highs = np.array(ci_highs)
    error_bar = np.array([means - ci_lows, ci_highs - means])
    error_bar = np.array([means - ci_lows, ci_highs - means])
    plt.rcParams["axes.formatter.useoffset"] = False
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
        f"/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments/RCS_next/ALL_RCS_{env_name}_mean_seeds_CI_normalised.png",
        dpi=300,
    )


# import os

# import matplotlib.pyplot as plt
# import numpy as np


def plot_algorithm_comparison(
    x_algo_paths,
    y_algo_paths,
    x_axis_algo_name,
    y_axis_algo_name,
    save_path,
    file_name,
    environments,
):
    """
    Plots a scatter plot comparing the episode returns of two algorithms across different environments.

    Parameters:
    -----------
    x_algo_paths : list of str
        A list of file paths to the numpy arrays containing the episode returns for the algorithm to be plotted on the x-axis (e.g., PPO).
    y_algo_paths : list of str
        A list of file paths to the numpy arrays containing the episode returns for the algorithm to be plotted on the y-axis (e.g., RND).
    x_axis_algo_name : str
        The name of the algorithm corresponding to the x-axis.
    y_axis_algo_name : str
        The name of the algorithm corresponding to the y-axis.
    save_path : str
        The directory path where the generated plot will be saved.
    file_name : str
        The name of the file to save the plot as, including the file extension (e.g., 'ppo_vs_rnd_plot.png').
    environments : list of str
        A list of environment names corresponding to the paths.

    Returns:
    --------
    None
        The function saves the scatter plot as an image file at the specified path.

    Example:
    --------
    plot_algorithm_comparison(x_algo_paths, y_algo_paths, 'PPO', 'Reward Combiner', './plots', 'comparison_plot.png', environments)
    """

    # Initialize lists to hold data and colors
    x_axis, y_axis = [], []
    color_map = plt.get_cmap("tab20")  # Using a colormap with many distinct colors
    colors = color_map(range(len(environments)))  # Assigning unique colors

    # Iterate over the provided paths and calculate means
    for i, (x_algo_path, y_algo_path) in enumerate(zip(x_algo_paths, y_algo_paths)):
        # Calculate the mean of the last 100 update steps, averaged over seeds
        x_mean = np.load(x_algo_path).mean(axis=-1)[-1]
        y_mean = np.load(y_algo_path).mean(axis=-1)[-1]
        x_axis.append(x_mean)
        y_axis.append(y_mean)

    # Create scatter plot with specified colors
    plt.figure(figsize=(12, 10))  # Increase figure size for better visibility
    plt.scatter(x_axis, y_axis, s=50, alpha=0.7, c=colors)  # Unique colors for each environment

    # Plot the y=x line for reference and add to legend
    min_val = min(min(x_axis), min(y_axis))
    max_val = max(max(x_axis), max(y_axis))
    (yx_line,) = plt.plot(
        [min_val, max_val], [min_val, max_val], linestyle="--", color="red", label="y=x"
    )

    # Set plot title and labels
    plt.xlabel(f"{x_axis_algo_name} Episode Returns")
    plt.ylabel(f"{y_axis_algo_name} Episode Returns")

    # Adjust axes limits for better visibility of low and high returns
    plt.xscale("symlog", linthresh=1)  # Corrected scale setting
    plt.yscale("symlog", linthresh=1)

    # Create a legend for the environment names
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=env_name,
            markerfacecolor=colors[i],
            markersize=10,
        )
        for i, env_name in enumerate(environments)
    ]
    handles.append(yx_line)  # Add the y=x line handle
    plt.legend(handles=handles, title="Environments", loc="center left", bbox_to_anchor=(1, 0.5))

    # Add grid
    plt.grid(True)

    # Construct the full save path
    full_save_path = os.path.join(save_path, file_name)

    # Save the plot to the specified path
    plt.savefig(
        full_save_path, bbox_inches="tight"
    )  # Save with tight bounding box to reduce whitespace
    plt.close()

    print(f"Plot saved to {full_save_path}")


def plot_histogram_with_error_bars(
    curious_paths, curious_algo_types, macro_env_types=["MiniGrid", "Brax", "MinAtar"]
):
    """
    Plots a grouped bar chart with error bars for multiple macro environments (MiniGrid, Brax, MinAtar).

    Parameters:
    - curious_paths: A list of lists of paths for each algorithm and environment.
    - curious_algo_types: A list of algorithm types (e.g., ['RND', 'PPO', 'BYOL-Explore']).
    - macro_env_types: A list of macro environments (MiniGrid, Brax, MinAtar environments).
    """
    # Indices for the macro environments based on your input
    env_indices = {
        "MiniGrid": slice(0, 6),  # First 6 environments
        "Brax": slice(6, 16),  # Next 10 environments
        "MinAtar": slice(16, 20),  # Last 4 environments
    }

    # Initialize lists for means and confidence intervals
    means_per_env = []
    ci_lows_per_env = []
    ci_highs_per_env = []

    # Loop over each macro environment type
    for macro_env in macro_env_types:
        env_slice = env_indices[macro_env]
        # Gather paths corresponding to this macro environment
        paths_for_macro_env = [paths[env_slice] for paths in curious_paths]

        # We pass paths for each algorithm separately to the get_normalise_cis_macro_env
        means = []
        ci_lows = []
        ci_highs = []
        for algo_paths in paths_for_macro_env:
            mean, ci_low, ci_high = get_normalise_cis_macro_env(algo_paths)
            means.append(mean)
            ci_lows.append(ci_low)
            ci_highs.append(ci_high)

        # Store the results for this macro environment
        means_per_env.append(np.array(means))
        ci_lows_per_env.append(np.array(ci_lows))
        ci_highs_per_env.append(np.array(ci_highs))

    means_per_env = np.array(means_per_env)
    ci_lows_per_env = np.array(ci_lows_per_env)
    ci_highs_per_env = np.array(ci_highs_per_env)

    bar_width = 0.2
    x = np.arange(len(curious_algo_types))  # Algorithms on the x-axis

    plt.figure(figsize=(12, 6))

    # Plot bars for each macro environment type
    for i, macro_env in enumerate(macro_env_types):
        means = means_per_env[i]
        ci_lows = ci_lows_per_env[i]
        ci_highs = ci_highs_per_env[i]

        lower_errors = means - ci_lows
        upper_errors = ci_highs - means
        error_bar = [lower_errors, upper_errors]

        # Plotting grouped bars for each macro environment
        plt.bar(
            x + i * bar_width,
            means,  # Mean for each macro env group
            bar_width,
            yerr=error_bar,  # Error bars
            capsize=5,
            label=macro_env,
        )

    plt.xlabel("Curious Algorithm", fontsize=12, fontweight="bold")
    plt.ylabel("Normalised Mean Episode Return", fontsize=12, fontweight="bold")
    plt.xticks(
        x + (len(macro_env_types) - 1) * bar_width / 2,
        curious_algo_types,
        rotation=45,
        ha="right",
        fontsize=10,
    )
    plt.yticks(fontsize=10)

    plt.legend(
        title="Macro Environments",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        fontsize=10,
    )
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(
        "/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments/images/all_macro_envs_mean_seeds_CI_normalised.png",
        dpi=300,
    )
    plt.show()


def plot_grouped_histogram(
    curious_paths, curious_algo_types, macro_env_types=["MiniGrid", "Brax", "MinAtar"]
):
    """
    Plots a grouped bar chart where each macro-environment has bars for the different algorithms side-by-side.

    Parameters:
    - curious_paths: A list of lists of paths for each algorithm and environment.
    - curious_algo_types: A list of algorithm types (e.g., ['RND', 'PPO', 'BYOL-Explore']).
    - macro_env_types: A list of macro environments (MiniGrid, Brax, MinAtar environments).
    """
    # Indices for the macro environments based on your input
    env_indices = {
        "MiniGrid": slice(0, 6),  # First 6 environments
        "Brax": slice(6, 16),  # Next 10 environments
        "MinAtar": slice(16, 20),  # Last 4 environments
    }

    # Initialize lists for means and confidence intervals
    means_per_env = []
    ci_lows_per_env = []
    ci_highs_per_env = []

    # Loop over each macro environment type
    for macro_env in macro_env_types:
        env_slice = env_indices[macro_env]
        # Gather paths corresponding to this macro environment
        paths_for_macro_env = [paths[env_slice] for paths in curious_paths]

        # We pass paths for each algorithm separately to the get_normalise_cis_macro_env
        means = []
        ci_lows = []
        ci_highs = []
        for algo_paths in paths_for_macro_env:
            mean, ci_low, ci_high = get_normalise_cis_macro_env(algo_paths)
            means.append(mean)
            ci_lows.append(ci_low)
            ci_highs.append(ci_high)

        # Store the results for this macro environment
        means_per_env.append(np.array(means))
        ci_lows_per_env.append(np.array(ci_lows))
        ci_highs_per_env.append(np.array(ci_highs))

    # Convert lists to arrays for easy manipulation
    means_per_env = np.array(means_per_env)
    ci_lows_per_env = np.array(ci_lows_per_env)
    ci_highs_per_env = np.array(ci_highs_per_env)

    # Bar width and position setup
    n_macro_envs = len(macro_env_types)
    n_algo = len(curious_algo_types)
    bar_width = 0.2
    x = np.arange(n_macro_envs)  # One group for each macro environment

    plt.figure(figsize=(12, 6))

    # Plot bars for each algorithm side-by-side for each macro environment
    for i, algo in enumerate(curious_algo_types):
        means = means_per_env[:, i]
        ci_lows = ci_lows_per_env[:, i]
        ci_highs = ci_highs_per_env[:, i]

        lower_errors = means - ci_lows
        upper_errors = ci_highs - means
        error_bar = [lower_errors, upper_errors]

        # Shift each set of bars for each algorithm next to each other for each macro environment
        plt.bar(
            x + i * bar_width,
            means,  # Mean for each algorithm group
            bar_width,
            yerr=error_bar,  # Error bars
            capsize=5,
            label=algo,
        )

    plt.xlabel("Macro Environments", fontsize=12, fontweight="bold")
    plt.ylabel("Normalised Mean Episode Return", fontsize=12, fontweight="bold")
    plt.xticks(
        x + (n_algo - 1) * bar_width / 2, macro_env_types, rotation=45, ha="right", fontsize=10
    )
    plt.yticks(fontsize=10)

    plt.legend(
        title="Curious Algorithms",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        fontsize=10,
    )
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(
        "/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments/images/grouped_algos_by_macro_env_100.png",
        dpi=300,
    )
    plt.show()


def get_experiment_directories(base_path, names, step_intervals, env_name):
    """
    Generate full paths to experiment directories.

    Args:
        base_path (str): Base directory containing experiments
        names (list): List of algorithm names
        step_intervals (list): List of step intervals
        env_name (str): Environment name

    Returns:
        dict: Directories organized by step interval
    """
    directories = {}
    for step_int in step_intervals:
        step_dirs = []
        for name in names:
            full_dir = os.path.join(base_path, f"{name}_{step_int}", env_name)
            step_dirs.append(full_dir)
        directories[step_int] = step_dirs
    return directories


def plot_final_episode_returns(directories, save_location, file_name):
    """
    Plot final episode returns with 95% confidence intervals for each step interval.

    Args:
        directories (dict): Directories containing experiment results, organized by step interval
        save_location (str): Directory to save plots
        file_name (str): Base filename for saved plots
    """
    # Set up the plotting style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    # Process each step interval separately
    for step_interval, dir_paths in directories.items():
        # Prepare data for plotting
        final_returns_data = []
        labels = []

        # Load and process data from each directory
        for dir_path in dir_paths:
            # Extract algorithm name
            parts = os.path.basename(os.path.dirname(dir_path)).split("_")
            algo_name = "_".join(parts[:-1])
            label = algo_name

            # Load numpy array
            metric_path = os.path.join(dir_path, "metric_seeds_episode_return.npy")
            if not os.path.exists(metric_path):
                print(f"Warning: {metric_path} not found. Skipping.")
                continue

            returns_data = np.load(metric_path)

            # Get final episode returns (last row of the array)
            final_returns = returns_data[-1, :]

            # Compute bootstrap confidence interval
            ci = bootstrap(
                (final_returns,),
                np.mean,
                confidence_level=0.95,
                method="percentile",
                n_resamples=10000,
            )

            final_returns_data.append(
                {
                    "mean": np.mean(final_returns),
                    "ci_low": ci.confidence_interval.low,
                    "ci_high": ci.confidence_interval.high,
                }
            )

            labels.append(label)

        # Create final returns plot for this step interval
        plt.figure(figsize=(10, 6))

        # Prepare data for plotting
        means = [d["mean"] for d in final_returns_data]
        ci_lows = [d["ci_low"] for d in final_returns_data]
        ci_highs = [d["ci_high"] for d in final_returns_data]

        # Plot error bars
        plt.errorbar(
            range(len(labels)),
            means,
            yerr=[np.array(means) - np.array(ci_lows), np.array(ci_highs) - np.array(means)],
            fmt="o",  # Circular markers
            capsize=5,  # Cap width for error bars
            capthick=1.5,  # Cap thickness
            ecolor="black",  # Error bar color
            markerfacecolor="blue",  # Marker fill color
            markeredgecolor="black",  # Marker edge color
            markersize=10,  # Marker size
            elinewidth=1.5,  # Error bar line width
        )

        plt.xlabel("Algorithms")
        plt.ylabel("Final Episode Return")
        plt.title(f"Final Episode Returns (Step Interval: {step_interval})")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.tight_layout()

        # Ensure save directory exists
        os.makedirs(save_location, exist_ok=True)

        # Save the plot
        plt.savefig(
            os.path.join(save_location, f"{file_name}_step_{step_interval}_final_returns.png"),
            dpi=300,
        )
        plt.close()


# name_to_normalises=["DELAY_RC_CNN_EPISODE","DELAY_RC_RNN"]
# step_intervals=[1,3,10,20,30,40]
# for name_to_normalise in name_to_normalises:
#     for step_int in step_intervals:
#         save_episode_return(
#             f"/home/batsi/Documents/Masters/MetaLearnCuriosity/{name_to_normalise}_Breakout-MinAtar_{step_int}_Breakout-MinAtar_flax-checkpoints_v0",
#             "/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments",
#             f"{name_to_normalise}_{step_int}",
#             "Breakout-MinAtar",
#         )


# Example usage
base_path = "/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments"
names = [
    "DELAY_RC_CNN",
    "TIMED_DELAY_RC_CNN",
    "DELAY_RC_RNN",
    "TIMED_DELAY_RC_RNN",
    "delayed_byol",
    "delayed_rnd",
    "minatar_baseline_ppo",
]
step_intervals = [1, 40]
env_name = "Breakout-MinAtar"

# Get directories
experiment_dirs = get_experiment_directories(base_path, names, step_intervals, env_name)

# Plot results
save_location = "/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments/images/Breakout-MinAtar"

# Plot results
plot_final_episode_returns(experiment_dirs, save_location, "breakout_final_returns")
