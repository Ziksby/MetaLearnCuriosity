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


# agent_type= "byol"
# plot_distribution(agent_type,path)


def plot_w_bootstrapped_CI(path):
    output = Restore(path)
    metric = output["metrics"]["returned_episode_returns"]
    # avg among the num of evns
    metric = jnp.mean(metric, axis=-1)

    # A 2d array of shape num_seeds, update step
    metric = metric.reshape(metric.shape[0], -1)

    # Transpose to make it (update steps, num_seeds)
    metric = metric.T

    means = []
    lows = []
    highs = []

    for i in range(len(metric)):
        timestep_values = (metric[i],)
        means.append(jnp.mean(metric[i]))
        ci = bootstrap(
            timestep_values, jnp.mean, confidence_level=0.95, method="percentile", n_resamples=1000
        )
        lows.append(ci.confidence_interval.low)
        highs.append(ci.confidence_interval.high)

    # Use Seaborn for a better style
    sns.set(style="whitegrid")

    # Create a Matplotlib figure and axis
    plt.figure(figsize=(10, 6))

    # Plot the mean values
    plt.plot(means, label="Mean Values")

    # Fill between the confidence intervals
    plt.fill_between(range(len(means)), lows, highs, alpha=0.2, label="95% CI")

    # Add labels and title
    plt.xlabel("Update Step")
    plt.ylabel("Mean Episode Return")

    # Show the plot
    plt.legend()
    plt.savefig(f"{path}/mean_seeds.png")


path = "MLC_logs/flax_ckpt/Empty-misc/byol_lite_empty_30"
plot_w_bootstrapped_CI(path)
