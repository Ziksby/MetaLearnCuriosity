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


def save_norm_int_rewards(path_to_extract, path_to_save, type_agent, env_name):
    start_time = time.time()
    output = Restore(path_to_extract)
    metric = output["norm_int_reward"]
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
    metric_file = os.path.join(save_path, "norm_int_reward.npy")
    means_file = os.path.join(save_path, "means_norm_int_reward.npy")
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


def save_norm_ext_reward(path_to_extract, path_to_save, type_agent, env_name):
    start_time = time.time()
    output = Restore(path_to_extract)
    metric = output["norm_ext_reward"]
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
    metric_file = os.path.join(save_path, "metric_seeds_norm_ext_reward.npy")
    means_file = os.path.join(save_path, "means_norm_ext_reward.npy")
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
    metrics = metrics[-1]

    ci = bootstrap(
        (metrics,),
        jnp.mean,
        confidence_level=0.95,
        method="percentile",
        n_resamples=10_000,
    )
    return metrics.mean(), ci.confidence_interval.low, ci.confidence_interval.high


def plot_error_bars_env(curious_paths, env_name: str, curious_algo_types, save_name, use_log=False):
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
    # error_bar = np.array([means - ci_lows, ci_highs - means])

    plt.rcParams["axes.formatter.useoffset"] = False
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create error bar plot
    # bars = ax.errorbar(
    #     curious_algo_types,
    #     means,
    #     yerr=error_bar,
    #     fmt="o",
    #     capsize=5,
    #     capthick=2,
    #     elinewidth=1.5,
    #     color="blue",
    #     markersize=8,
    # )

    # Create a secondary box on the side with mean values
    text_box = "\n".join([f"{algo}: {mean:.2e}" for algo, mean in zip(curious_algo_types, means)])
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    ax.text(
        1.05,
        0.5,
        text_box,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="center",
        bbox=props,
    )

    ax.set_xlabel("Curious Algorithm", fontsize=12, fontweight="bold", fontname="DejaVu Sans")
    ax.set_ylabel(
        "Mean Episode Return" if not use_log else "Mean Episode Return (Log Scale)",
        fontsize=12,
        fontweight="bold",
        fontname="DejaVu Sans",
    )

    if use_log:
        ax.set_yscale("log")  # Use log scale on y-axis if enabled

    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xticks(range(len(curious_algo_types)))
    ax.set_xticklabels(
        curious_algo_types, rotation=45, ha="right", fontsize=10, fontname="DejaVu Sans"
    )
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    prefix = "log_" if use_log else ""
    plt.savefig(
        f"/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments/images/{prefix}{save_name}_{env_name}.png",
        dpi=300,
    )
    plt.close()


def plot_training_curves_env(curious_paths, env_name: str, curious_algo_types, save_name):
    plt.figure(figsize=(10, 6))

    for path, algo_type in zip(curious_paths, curious_algo_types):
        # Load full training history
        metrics = np.load(path)
        num_steps = metrics.shape[0]

        # Calculate mean across seeds for each timestep
        means = np.mean(metrics, axis=1)

        # Calculate CIs for each timestep
        ci_lows = []
        ci_highs = []

        for step in range(num_steps):
            step_metrics = metrics[step]
            ci = bootstrap(
                (step_metrics,),
                jnp.mean,
                confidence_level=0.95,
                method="percentile",
                n_resamples=10_000,
            )
            ci_lows.append(ci.confidence_interval.low)
            ci_highs.append(ci.confidence_interval.high)

        ci_lows = np.array(ci_lows)
        ci_highs = np.array(ci_highs)

        # Create x-axis for steps
        steps = np.arange(num_steps)

        # Plot mean line and shaded CI region
        plt.plot(steps, means, label=algo_type, linewidth=2)
        plt.fill_between(steps, ci_lows, ci_highs, alpha=0.2)

    plt.xlabel("Update Steps", fontsize=12, fontweight="bold", fontname="DejaVu Sans")
    plt.ylabel("Mean Episode Return", fontsize=12, fontweight="bold", fontname="DejaVu Sans")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(fontsize=10, fontname="DejaVu Sans")
    plt.yticks(fontsize=10, fontname="DejaVu Sans")

    plt.tight_layout()
    plt.savefig(
        f"/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments/images/training_curves_{save_name}_{env_name}.png",
        dpi=300,
    )
    plt.close()


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


def plot_int_lambdas_for_seed(
    path_to_saved_metrics: str,
    type_agent: str,
    env_name: str,
    path_to_save: str,
    seed_selection: str = "highest",
    filename: str = None,
):
    """
    Plot int_lambdas for a specified seed or the best-performing seed based on saved episode returns.

    Parameters:
    -----------
    path_to_saved_metrics : str
        Path to the directory containing the saved metric files
    type_agent : str
        Name referencing the agent (used in file paths and plot legend, but not labeled as 'agent')
    env_name : str
        Name of the environment (used internally for file loading, not displayed in plot)
    path_to_save : str
        Path to save the generated plot
    seed_selection : str or int, optional (default='highest')
        Can be 'highest' to select the seed with the highest last timestep return,
        or an integer representing the specific seed index (0-based)
    filename : str, optional
        Custom filename for the saved plot

    Returns:
    --------
    int : Seed index used for plotting
    """
    # Construct paths to metric files
    base_path = os.path.join(path_to_saved_metrics, type_agent, env_name)

    # Load episode returns
    episode_returns_file = os.path.join(base_path, "metric_seeds_episode_return.npy")
    episode_returns = np.load(episode_returns_file)

    # Determine seed selection
    if isinstance(seed_selection, str) and seed_selection.lower() == "highest":
        # Find the seed with the highest value at the last timestep
        seed_index = np.argmax(episode_returns[-1])
    elif isinstance(seed_selection, int):
        # Use the specified seed index (0-based)
        seed_index = seed_selection
    else:
        raise ValueError("seed_selection must be 'highest' or an integer")

    # Load int_lambdas
    int_lambdas_file = os.path.join(base_path, "int_lambda_seeds_episode_return.npy")
    int_lambdas = np.load(int_lambdas_file)

    # Select the specific seed's int_lambdas
    seed_int_lambdas = int_lambdas[:, seed_index]

    # Create a clean, publication-quality plot
    plt.figure(figsize=(6, 4), dpi=300)

    # Plot the intrinsic lambda values
    # Legend references the agent by type_agent, but doesn't use the word "agent".
    plt.plot(seed_int_lambdas, color="dodgerblue", linewidth=2, label=f"{type_agent}")

    # Axis labels
    plt.xlabel("Update Steps", fontsize=12)
    plt.ylabel("Intrinsic Lambda", fontsize=12)

    # Add a legend without the word 'agent'
    plt.legend(frameon=False, fontsize=10, loc="best")

    # Grid and aesthetic touches
    plt.grid(True, linestyle="--", linewidth=0.5, color="lightgray")
    plt.tight_layout()
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Determine save path and filename
    save_path = os.path.join(path_to_save, type_agent, env_name)
    os.makedirs(save_path, exist_ok=True)

    if filename is None:
        filename = f"int_lambdas_seed_{seed_index}.png"

    full_save_path = os.path.join(save_path, filename)
    plt.savefig(full_save_path, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {full_save_path}")
    print(f"Seed index used: {seed_index}")

    return seed_index


def plot_norm_int_rew_for_seed(
    path_to_saved_metrics: str,
    type_agent: str,
    env_name: str,
    path_to_save: str,
    seed_selection: str = "highest",
    filename: str = None,
):
    """
    Plot int_lambdas for a specified seed or the best-performing seed based on saved episode returns.

    Parameters:
    -----------
    path_to_saved_metrics : str
        Path to the directory containing the saved metric files
    type_agent : str
        Name referencing the agent (used in file paths and plot legend, but not labeled as 'agent')
    env_name : str
        Name of the environment (used internally for file loading, not displayed in plot)
    path_to_save : str
        Path to save the generated plot
    seed_selection : str or int, optional (default='highest')
        Can be 'highest' to select the seed with the highest last timestep return,
        or an integer representing the specific seed index (0-based)
    filename : str, optional
        Custom filename for the saved plot

    Returns:
    --------
    int : Seed index used for plotting
    """
    # Construct paths to metric files
    base_path = os.path.join(path_to_saved_metrics, type_agent, env_name)

    # Load episode returns
    episode_returns_file = os.path.join(base_path, "metric_seeds_episode_return.npy")
    episode_returns = np.load(episode_returns_file)

    # Determine seed selection
    if isinstance(seed_selection, str) and seed_selection.lower() == "highest":
        # Find the seed with the highest value at the last timestep
        seed_index = np.argmax(episode_returns[-1])
    elif isinstance(seed_selection, int):
        # Use the specified seed index (0-based)
        seed_index = seed_selection
    else:
        raise ValueError("seed_selection must be 'highest' or an integer")

    # Load int_lambdas
    int_lambdas_file = os.path.join(base_path, "norm_int_reward.npy")
    int_lambdas = np.load(int_lambdas_file)

    # Select the specific seed's int_lambdas
    seed_int_lambdas = int_lambdas[:, seed_index]

    # Create a clean, publication-quality plot
    plt.figure(figsize=(6, 4), dpi=300)

    # Plot the intrinsic lambda values
    # Legend references the agent by type_agent, but doesn't use the word "agent".
    plt.plot(seed_int_lambdas, color="dodgerblue", linewidth=2, label=f"{type_agent}")

    # Axis labels
    plt.xlabel("Update Steps", fontsize=12)
    plt.ylabel("Normalised Intrinsic Rewards", fontsize=12)

    # Add a legend without the word 'agent'
    plt.legend(frameon=False, fontsize=10, loc="best")

    # Grid and aesthetic touches
    plt.grid(True, linestyle="--", linewidth=0.5, color="lightgray")
    plt.tight_layout()
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Determine save path and filename
    save_path = os.path.join(path_to_save, type_agent, env_name)
    os.makedirs(save_path, exist_ok=True)

    if filename is None:
        filename = f"norm_int_rew_seed_{seed_index}.png"

    full_save_path = os.path.join(save_path, filename)
    plt.savefig(full_save_path, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {full_save_path}")
    print(f"Seed index used: {seed_index}")

    return seed_index


def plot_norm_ext_for_seed(
    path_to_saved_metrics: str,
    type_agent: str,
    env_name: str,
    path_to_save: str,
    seed_selection: str = "highest",
    filename: str = None,
):
    """
    Plot int_lambdas for a specified seed or the best-performing seed based on saved episode returns.

    Parameters:
    -----------
    path_to_saved_metrics : str
        Path to the directory containing the saved metric files
    type_agent : str
        Name referencing the agent (used in file paths and plot legend, but not labeled as 'agent')
    env_name : str
        Name of the environment (used internally for file loading, not displayed in plot)
    path_to_save : str
        Path to save the generated plot
    seed_selection : str or int, optional (default='highest')
        Can be 'highest' to select the seed with the highest last timestep return,
        or an integer representing the specific seed index (0-based)
    filename : str, optional
        Custom filename for the saved plot

    Returns:
    --------
    int : Seed index used for plotting
    """
    # Construct paths to metric files
    base_path = os.path.join(path_to_saved_metrics, type_agent, env_name)

    # Load episode returns
    episode_returns_file = os.path.join(base_path, "metric_seeds_episode_return.npy")
    episode_returns = np.load(episode_returns_file)

    # Determine seed selection
    if isinstance(seed_selection, str) and seed_selection.lower() == "highest":
        # Find the seed with the highest value at the last timestep
        seed_index = np.argmax(episode_returns[-1])
    elif isinstance(seed_selection, int):
        # Use the specified seed index (0-based)
        seed_index = seed_selection
    else:
        raise ValueError("seed_selection must be 'highest' or an integer")

    # Load int_lambdas
    int_lambdas_file = os.path.join(base_path, "norm_ext_reward.npy")
    int_lambdas = np.load(int_lambdas_file)

    # Select the specific seed's int_lambdas
    seed_int_lambdas = int_lambdas[:, seed_index]

    # Create a clean, publication-quality plot
    plt.figure(figsize=(6, 4), dpi=300)

    # Plot the intrinsic lambda values
    # Legend references the agent by type_agent, but doesn't use the word "agent".
    plt.plot(seed_int_lambdas, color="dodgerblue", linewidth=2, label=f"{type_agent}")

    # Axis labels
    plt.xlabel("Update Steps", fontsize=12)
    plt.ylabel("Normalised Intrinsic Rewards", fontsize=12)

    # Add a legend without the word 'agent'
    plt.legend(frameon=False, fontsize=10, loc="best")

    # Grid and aesthetic touches
    plt.grid(True, linestyle="--", linewidth=0.5, color="lightgray")
    plt.tight_layout()
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Determine save path and filename
    save_path = os.path.join(path_to_save, type_agent, env_name)
    os.makedirs(save_path, exist_ok=True)

    if filename is None:
        filename = f"norm_int_rew_seed_{seed_index}.png"

    full_save_path = os.path.join(save_path, filename)
    plt.savefig(full_save_path, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {full_save_path}")
    print(f"Seed index used: {seed_index}")

    return seed_index


def create_path_to_file(base_dir, names, env_names, filename):
    """
    Creates directories leading to a specific file for a given base directory, names, and env_names.

    Parameters:
        base_dir (str): The base directory where new directories will be created.
        names (list): A list of folder names (e.g., experiment names) to append to the base directory.
        env_names (list): A list of environment names to create subdirectories for each name.
        filename (str): The name of the file to create directories up to.

    Returns:
        list: A list of full paths to the files.
    """
    file_paths = []

    for name in names:
        for env_name in env_names:
            # Construct the full path
            full_dir = os.path.join(base_dir, name, env_name)

            # Ensure the directories exist
            os.makedirs(full_dir, exist_ok=True)

            # Construct the full path to the file
            file_path = os.path.join(full_dir, filename)
            file_paths.append(file_path)

    return file_paths


def plot_norm_int_rewards(file_path, env_name, save_path):
    data = np.load(file_path)

    # Transpose the array
    data_transposed = data.T

    # Extract the 15th index (14 in zero-based indexing)
    reward_data = data_transposed[15]

    # Create the plot
    plt.figure(figsize=(8, 6))  # Set figure size

    # Plot the data
    plt.plot(reward_data, label="BYOL_RC_RNN", linewidth=2, color="blue")

    # Add titles and labels
    # plt.title(f"Normalized Intrinsic Reward in {env_name}", fontsize=14)
    plt.xlabel("Update Step", fontsize=12)
    plt.ylabel("Normalized Intrinsic Reward", fontsize=12)

    # Add a grid for better readability
    plt.grid(alpha=0.3)

    # Add a legend
    plt.legend(fontsize=12)

    # Customize the ticks
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Save the plot (optional)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{env_name}_intrinsic_reward_plot.png", dpi=300)


def create_scatter_plots(algorithm_name, env_name, base_path, seed=None, save_path=None):
    """
    Create scatter plots of lambda values vs normalized rewards.

    Args:
        algorithm_name (str): Name of the algorithm (e.g., 'RC-Action')
        env_name (str): Name of the environment (e.g., 'MiniGrid-EmptyRandom-16x16')
        base_path (str): Base path to the experiments directory
        seed (int, optional): Specific seed to plot. If None, uses seed with highest final episode return
        save_path (str, optional): Path to save the plots. If None, uses the experiment directory
    """
    # Construct full path
    exp_path = os.path.join(base_path, algorithm_name, env_name)

    # Load data
    means_norm_ext_reward = np.load(os.path.join(exp_path, "metric_seeds_norm_ext_reward.npy"))
    means_int_lambda = np.load(os.path.join(exp_path, "int_lambda_seeds_episode_return.npy"))
    means_norm_int_reward = np.load(os.path.join(exp_path, "norm_int_reward.npy"))
    means_episode_return = np.load(os.path.join(exp_path, "metric_seeds_episode_return.npy"))

    # Print shapes
    print(f"Shape of arrays: {means_norm_ext_reward.shape}")

    # Determine seed if not specified
    if seed is None:
        best_seed = np.argmax(means_episode_return[-1])
        seed = best_seed
        print(f"Using best performing seed: {seed}")
    else:
        print(f"Using specified seed: {seed}")

    # Set save path if not specified
    if save_path is None:
        save_path = exp_path

    # Set style for publication-quality plots
    plt.style.use("seaborn-v0_8-whitegrid")

    # Create and save intrinsic reward plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        means_norm_int_reward[:, seed], means_int_lambda[:, seed], alpha=0.6, color="#2878B5"
    )
    plt.xlabel("Normalized Intrinsic Reward")
    plt.ylabel("$\lambda$ Values")  # noqa
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f"THESIS_{algorithm_name}_lambda_vs_int_reward.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Create and save extrinsic reward plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        means_norm_ext_reward[:, seed], means_int_lambda[:, seed], alpha=0.6, color="#2878B5"
    )
    plt.xlabel("Normalized Extrinsic Reward")
    plt.ylabel("$\lambda$ Values")  # noqa
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f"THESIS_{algorithm_name}_lambda_vs_ext_reward.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Create and save episode return plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        means_episode_return[:, seed], means_int_lambda[:, seed], alpha=0.6, color="#2878B5"
    )
    plt.xlabel("Episode Return")
    plt.ylabel("$\lambda$ Values")  # noqa
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f"THESIS_{algorithm_name}_lambda_vs_episode_return.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Plots saved in {save_path}")


# def create_lambda_line_plots(algorithm_name, env_name, base_path, seed=None, save_path=None):
#     """
#     Create line plots of lambda values vs normalized rewards over update steps.

#     Generates both linear and log scale plots for comparison, except for individual metric plots.
#     """
#     # Construct full path
#     exp_path = os.path.join(base_path, algorithm_name, env_name)

#     # Load data
#     means_norm_ext_reward = np.load(os.path.join(exp_path, 'metric_seeds_norm_ext_reward.npy'))
#     means_int_lambda = np.load(os.path.join(exp_path, 'int_lambda_seeds_episode_return.npy'))
#     means_norm_int_reward = np.load(os.path.join(exp_path, 'norm_int_reward.npy'))
#     means_episode_return = np.load(os.path.join(exp_path, 'metric_seeds_episode_return.npy'))

#     # Determine seed if not specified
#     if seed is None:
#         best_seed = np.argmax(means_episode_return[-1])
#         seed = best_seed

#     # Set save path if not specified
#     if save_path is None:
#         save_path = exp_path

#     # Set style for publication-quality plots
#     plt.style.use('seaborn-v0_8-whitegrid')

#     # Get update steps
#     update_steps = np.arange(means_int_lambda.shape[0])

#     # List of comparisons for both linear and log plots
#     comparisons = [
#         ("lambda_vs_ext_reward", [means_int_lambda[:, seed], means_norm_ext_reward[:, seed]],
#          [r'$\lambda$ value', 'Normalized Extrinsic Reward']),
#         ("lambda_vs_int_reward", [means_int_lambda[:, seed], means_norm_int_reward[:, seed]],
#          [r'$\lambda$ value', 'Normalized Intrinsic Reward']),
#         ("lambda_vs_rewards", [means_int_lambda[:, seed], means_norm_ext_reward[:, seed], means_norm_int_reward[:, seed]],
#          [r'$\lambda$ value', 'Normalized Extrinsic Reward', 'Normalized Intrinsic Reward'])
#     ]

#     for name, data, labels in comparisons:
#         # Linear Scale Plot
#         plt.figure(figsize=(8, 6))
#         for d, label in zip(data, labels):
#             plt.plot(update_steps, d, label=label)
#         plt.xlabel('Update Step')
#         plt.ylabel('Metric Values')
#         plt.legend(frameon=False)
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_path, f'{algorithm_name}_{env_name}_seed{seed}_{name}_linear.png'),
#                     dpi=300, bbox_inches='tight')
#         plt.close()

#         # Log Scale Plot
#         plt.figure(figsize=(8, 6))
#         for d, label in zip(data, labels):
#             plt.plot(update_steps, d, label=label)
#         plt.xlabel('Update Step')
#         plt.ylabel('Log(Metric Values)')
#         plt.yscale('log')
#         plt.legend(frameon=False)
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_path, f'{algorithm_name}_{env_name}_seed{seed}_{name}_log.png'),
#                     dpi=300, bbox_inches='tight')
#         plt.close()


def create_lambda_line_plots(algorithm_name, env_name, base_path, seed=None, save_path=None):
    """
    Create line plots of lambda values vs normalised rewards over update steps.
    Uses twinx() for plots with multiple metrics to ensure readability.
    """
    exp_path = os.path.join(base_path, algorithm_name, env_name)

    # Load data
    means_norm_ext_reward = np.load(os.path.join(exp_path, "metric_seeds_norm_ext_reward.npy"))
    means_int_lambda = np.load(os.path.join(exp_path, "int_lambda_seeds_episode_return.npy"))
    means_norm_int_reward = np.load(os.path.join(exp_path, "norm_int_reward.npy"))
    means_episode_return = np.load(os.path.join(exp_path, "metric_seeds_episode_return.npy"))

    if seed is None:
        best_seed = np.argmax(means_episode_return[-1])
        seed = best_seed

    if save_path is None:
        save_path = exp_path

    plt.style.use("seaborn-v0_8-darkgrid")
    update_steps = np.arange(means_int_lambda.shape[0])

    # Individual metric plots (no changes here)
    individual_metrics = [
        ("lambda", means_int_lambda[:, seed], r"$\lambda$ value"),
        ("ext_reward", means_norm_ext_reward[:, seed], "Normalised Extrinsic Reward"),
        ("int_reward", means_norm_int_reward[:, seed], "Normalised Intrinsic Reward"),
    ]

    colours = ["#1f77b4", "#9467bd", "#2ca02c"]

    for (name, data, label), colour in zip(individual_metrics, colours):
        plt.figure(figsize=(8, 6))
        plt.plot(update_steps, data, label=label, color=colour)
        plt.xlabel("Update Step")
        plt.ylabel(label)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_path, f"{algorithm_name}_{env_name}_seed{seed}_{name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Dual-axis plots using twinx()
    comparisons = [
        (
            "lambda_vs_ext_reward",
            means_int_lambda[:, seed],
            means_norm_ext_reward[:, seed],
            r"$\lambda$ value",
            "Normalised Extrinsic Reward",
        ),
        (
            "lambda_vs_int_reward",
            means_int_lambda[:, seed],
            means_norm_int_reward[:, seed],
            r"$\lambda$ value",
            "Normalised Intrinsic Reward",
        ),
    ]

    for name, data1, data2, label1, label2 in comparisons:
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(update_steps, data1, label=label1, color=colours[0])
        ax1.set_xlabel("Update Step")
        ax1.set_ylabel(label1, color=colours[0])
        ax1.tick_params(axis="y", labelcolor=colours[0])

        ax2 = ax1.twinx()
        ax2.plot(update_steps, data2, label=label2, color=colours[1])
        ax2.set_ylabel(label2, color=colours[1])
        ax2.tick_params(axis="y", labelcolor=colours[1])

        fig.tight_layout()
        plt.savefig(
            os.path.join(save_path, f"{algorithm_name}_{env_name}_seed{seed}_{name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Triple-metric plot using twinx()
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(update_steps, means_int_lambda[:, seed], label=r"$\lambda$ value", color=colours[0])
    ax1.set_xlabel("Update Step")
    ax1.set_ylabel(r"$\lambda$ value", color=colours[0])
    ax1.tick_params(axis="y", labelcolor=colours[0])

    ax2 = ax1.twinx()
    ax2.plot(
        update_steps,
        means_norm_ext_reward[:, seed],
        label="Normalised Extrinsic Reward",
        color=colours[1],
    )
    ax2.set_ylabel("Normalised Extrinsic Reward", color=colours[1])
    ax2.tick_params(axis="y", labelcolor=colours[1])

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))  # Offset the third y-axis
    ax3.plot(
        update_steps,
        means_norm_int_reward[:, seed],
        label="Normalised Intrinsic Reward",
        color=colours[2],
    )
    ax3.set_ylabel("Normalised Intrinsic Reward", color=colours[2])
    ax3.tick_params(axis="y", labelcolor=colours[2])

    fig.tight_layout()
    plt.savefig(
        os.path.join(save_path, f"{algorithm_name}_{env_name}_seed{seed}_lambda_vs_rewards.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_lambda_3d_and_scatter(algorithm_name, env_name, base_path, seed=None, save_path=None):
    """
    Create 3D surface and 2D scatter plots of lambda values with normalized intrinsic and extrinsic rewards.

    Args:
        algorithm_name (str): Name of the algorithm (e.g., 'RC-Action')
        env_name (str): Name of the environment (e.g., 'MiniGrid-EmptyRandom-16x16')
        base_path (str): Base path to the experiments directory
        seed (int, optional): Specific seed to plot. If None, uses seed with highest final episode return
        save_path (str, optional): Path to save the plots. If None, uses the experiment directory
    """
    # Construct full path
    exp_path = os.path.join(base_path, algorithm_name, env_name)

    # Load data
    means_norm_ext_reward = np.load(os.path.join(exp_path, "metric_seeds_norm_ext_reward.npy"))
    means_int_lambda = np.load(os.path.join(exp_path, "int_lambda_seeds_episode_return.npy"))
    means_norm_int_reward = np.load(os.path.join(exp_path, "norm_int_reward.npy"))
    means_episode_return = np.load(os.path.join(exp_path, "metric_seeds_episode_return.npy"))

    # Print shapes
    print(f"Shape of arrays: {means_norm_ext_reward.shape}")

    # Determine seed if not specified
    if seed is None:
        best_seed = np.argmax(means_episode_return[-1])
        seed = best_seed
        print(f"Using best performing seed: {seed}")
    else:
        print(f"Using specified seed: {seed}")

    # Set save path if not specified
    if save_path is None:
        save_path = exp_path

    # Extract seed-specific data
    r_int = means_norm_int_reward[:, seed]  # X-axis (Intrinsic Reward)
    r_ext = means_norm_ext_reward[:, seed]  # Y-axis (Extrinsic Reward)
    lambda_int = means_int_lambda[:, seed]  # Z-axis / Color (Lambda Values)

    # ---- 3D Surface Plot ----
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(r_int, r_ext, lambda_int, cmap="viridis", edgecolor="none")

    ax.set_xlabel("Normalized Intrinsic Reward")
    ax.set_ylabel("Normalized Extrinsic Reward")
    ax.set_zlabel(r"$\lambda$ value")
    plt.tight_layout()

    # Save 3D surface plot
    save_filename_3d = f"{algorithm_name}_{env_name}_seed{seed}_lambda_3d.png"
    plt.savefig(os.path.join(save_path, save_filename_3d), dpi=300, bbox_inches="tight")
    plt.close()

    # ---- 2D Scatter Plot with Color Encoding ----
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(r_int, r_ext, c=lambda_int, cmap="viridis", alpha=0.8, edgecolors="k")
    plt.colorbar(scatter, label=r"$\lambda$ value")

    plt.xlabel("Normalized Intrinsic Reward")
    plt.ylabel("Normalized Extrinsic Reward")
    plt.tight_layout()

    # Save 2D scatter plot
    save_filename_2d = f"{algorithm_name}_{env_name}_seed{seed}_lambda_scatter.png"
    plt.savefig(os.path.join(save_path, save_filename_2d), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plots saved in {save_path} as {save_filename_3d} and {save_filename_2d}")


names = [
    "RC-Base",
    "RC-Action",
    "RC-NoInput",
    "RC-NoAction",
]  # "BYOL-Explore-Best-Lambda", "RND-Best-Lambda", "PPO-RNN", "BYOL-Unlock"

# r=Restore("/home/batsi/Documents/Masters/MetaLearnCuriosity/artefacts/BYOL_minigrid_MiniGrid-EmptyRandom-16x16_MiniGrid-EmptyRandom-16x16_flax-checkpoints_v1")
# # print(r["int_reward"].shape,r["norm_int_reward"].shape,r.keys())
# path_to_extract, path_to_save, type_agent, env_name
base = "/home/batsi/Documents/Masters/MetaLearnCuriosity/artefacts"
base_dir = "/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments"
# /home/batsi/Documents/Masters/MetaLearnCuriosity/artefacts/BYOL_minigrid_MiniGrid-EmptyRandom-16x16_FROM_UNLOCK_OPTIMAL_MiniGrid-EmptyRandom-16x16_flax-checkpoints_v0
name = "BYOL_minigrid"
envs = [
    # "MiniGrid-Empty-16x16",
    # "MiniGrid-Empty-8x8",
    # "MiniGrid-Empty-8x8",
    # "MiniGrid-Empty-5x5",
    # "MiniGrid-EmptyRandom-16x16",
    # "MiniGrid-EmptyRandom-8x8",
    # "MiniGrid-EmptyRandom-6x6",
    # "MiniGrid-EmptyRandom-5x5",
    "MiniGrid-DoorKey-8x8",
    # "MiniGrid-DoorKey-6x6",
    # "MiniGrid-DoorKey-5x5",
    # "MiniGrid-FourRooms",
    # "MiniGrid-MemoryS8",
    # "MiniGrid-MemoryS16",
    "MiniGrid-Unlock",
]
# envs=[ "MiniGrid-DoorKey-16x16","MiniGrid-UnlockPickUp","MiniGrid-BlockedUnlockPickUp"] #
# envs=["MiniGrid-EmptyRandom-16x16","MiniGrid-DoorKey-8x8", "MiniGrid-Unlock"]
envs = ["MiniGrid-DoorKey-8x8"]
env_name = envs[0]
# name = "80_GEN_DEFAULTS_ACTION_BYOL_RC_RNN"
# /home/batsi/Documents/Masters/MetaLearnCuriosity/artefacts/BYOL_minigrid_MiniGrid-EmptyRandom-16x16_FROM_UNLOCK_OPTIMAL_MiniGrid-EmptyRandom-16x16_flax-checkpoints_v0
# for env_name in envs:
#     path = f"{base}/{name}_{env_name}_FROM_RANDOM_OPTIMAL_{env_name}_flax-checkpoints_v0"
#     save_episode_return(path , "/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments","BYOL-EmptyRandom",env_name)
# #     save_int_lambdas(path , "/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments","RC-NoAction",env_name)
# #     save_norm_int_rewards(path , "/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments","RC-NoAction",env_name)
#     save_norm_ext_reward(path , "/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments","RC-Action",env_name)
create_lambda_line_plots(
    "RC-Action",
    "MiniGrid-Unlock",
    "/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments",
    save_path="/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments/images/ACTION_RC_RNN",
)
# Load the numpy array
# save_episode_return()
# fn="metric_seeds_episode_return.npy"
# c_ps=create_path_to_file(base_dir, names, envs, fn)
# # plot_training_curves_env(c_ps, env_name, names, f"RC-ablation-results-v3-{env_name}")
# plot_error_bars_env(c_ps, env_name, names, f"THESIS-ablation-action-{env_name}", use_log=True)
# create_scatter_plots("RC-Base",
#                      env_name,
#                      "/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments",
#                      )
# for env_name in envs:
#     plot_int_lambdas_for_seed("/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments",
#                             "ACTION_RC_RNN",
#                             env_name,
#                             "/home/batsi/Documents/Masters/MetaLearnCuriosity/MetaLearnCuriosity/experiments/images",
#                             'highest',
#                             f"action_rc_rnn_int_lambda_{env_name}.png",
#                             )
