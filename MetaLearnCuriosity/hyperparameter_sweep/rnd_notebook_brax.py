# Taken from:
# https://github.com/corl-team/xland-minigrid/blob/main/training/train_single_task.py

import gc
import os
import shutil
import time
from typing import Any, NamedTuple, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.jax_utils import replicate, unreplicate
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from scipy.stats import bootstrap

import wandb
from MetaLearnCuriosity.agents.nn import (
    BraxBYOLPredictor,
    BYOLTarget,
    CloseScannedRNN,
    OpenScannedRNN,
    PredictorNetwork,
    TargetNetwork,
)
from MetaLearnCuriosity.checkpoints import Save
from MetaLearnCuriosity.logger import WBLogger
from MetaLearnCuriosity.utils import (
    ObsNormParams,
    RNDNormIntReturnParams,
    RNDTransition,
    make_obs_gymnax_discrete,
    rnd_normalise_int_rewards,
    update_obs_norm_params,
)
from MetaLearnCuriosity.wrappers import (
    BraxGymnaxWrapper,
    ClipAction,
    DelayedReward,
    LogWrapper,
    NormalizeVecObservation,
    NormalizeVecReward,
    ProbabilisticReward,
    VecEnv,
)

jax.config.update("jax_threefry_partitionable", True)

key = jax.random.PRNGKey(76)

environments = [
    "ant",
    # "halfcheetah",
    # "hopper",
    # "humanoid",
    # "humanoidstandup",
    # "inverted_pendulum",
    # "inverted_double_pendulum",
    # "pusher",
    # "reacher",
    # "walker2d",
]
step_intervals = [1]
config = {
    "RUN_NAME": "rnd_delayed_brax",
    "SEED": 42,
    "NUM_SEEDS": 1,
    "LR": 3e-4,
    "PRED_LR": 1e-3,
    "NUM_ENVS": 2048,
    "NUM_STEPS": 10,  # unroll length
    "TOTAL_TIMESTEPS": 5e7,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 32,
    "GAMMA": 0.99,
    "INT_GAMMA": 0.99,
    "EMA_PARAMETER": 0.99,
    "GAE_LAMBDA": 0.95,
    "INT_LAMBDA": 0.1,  # 0.00021,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.0,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ANNEAL_LR": False,
    "ANNEAL_PRED_LR": False,
    "NORMALIZE_ENV": True,
    "DELAY_REWARDS": True,
    "STEP_INTERVAL": 0.98,
    "DEBUG": False,
    "REW_NORM_PARAMETER": 0.99,
}


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            actor_mean
        )
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


def make_config_env(config, env_name):
    config["ENV_NAME"] = env_name
    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    num_devices = jax.local_device_count()
    assert config["NUM_ENVS"] % num_devices == 0
    config["NUM_ENVS_PER_DEVICE"] = config["NUM_ENVS"] // num_devices
    config["TOTAL_TIMESTEPS_PER_DEVICE"] = config["TOTAL_TIMESTEPS"] // num_devices
    # config["EVAL_EPISODES_PER_DEVICE"] = config["EVAL_EPISODES"] // num_devices
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS_PER_DEVICE"] // config["NUM_STEPS"] // config["NUM_ENVS_PER_DEVICE"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS_PER_DEVICE"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["TRAINING_HORIZON"] = (
        config["TOTAL_TIMESTEPS_PER_DEVICE"] // config["NUM_ENVS_PER_DEVICE"]
    )
    assert config["NUM_ENVS_PER_DEVICE"] >= 4
    config["UPDATE_PROPORTION"] = 4 / config["NUM_ENVS_PER_DEVICE"]

    env = LogWrapper(env)
    env = ClipAction(env)
    if config["DELAY_REWARDS"]:
        env = DelayedReward(env, config["STEP_INTERVAL"])
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    return config, env, env_params


def make_train(rng):
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # def pred_linear_schedule(count):
    #     frac = (
    #         1.0
    #         - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
    #         / config["NUM_UPDATES"]
    #     )
    #     return config["PRED_LR"] * frac

    # INIT NETWORK
    network = ActorCritic(env.action_space(env_params).shape[0], activation=config["ACTIVATION"])
    target = TargetNetwork(256)
    predictor = PredictorNetwork(256)

    rng, _rng = jax.random.split(rng)
    rng, _pred_rng = jax.random.split(rng)
    rng, _tar_rng = jax.random.split(rng)
    rng, _init_obs_rng = jax.random.split(rng)

    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params = network.init(_rng, init_x)
    target_params = target.init(_tar_rng, init_x)
    pred_params = predictor.init(_pred_rng, init_x)

    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    pred_tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["PRED_LR"], eps=1e-5),
    )

    predictor_state = TrainState.create(apply_fn=predictor.apply, params=pred_params, tx=pred_tx)

    rng = jax.random.split(rng, jax.local_device_count())

    return rng, train_state, predictor_state, target_params, _init_obs_rng


def train(rng, train_state, pred_state, target_params, init_obs_rng):

    # INIT OBS NORM PARAMS:
    random_rollout = make_obs_gymnax_discrete(
        config["NUM_ENVS_PER_DEVICE"], env, env_params, config["NUM_STEPS"]
    )

    # Obs will be in shape: num_steps, num_envs, obs.shape
    init_obs = random_rollout(init_obs_rng)
    init_obs = init_obs.reshape(
        -1, init_obs.shape[-1]
    )  # reshape it to num_envs*num_steps, obs.shape

    init_mean_obs = jnp.zeros(init_obs.shape[-1])
    init_var_obs = jnp.ones(init_obs.shape[-1])
    init_obs_count = 1e-4

    init_obs_norm_params = ObsNormParams(init_obs_count, init_mean_obs, init_var_obs)
    rnd_int_return_norm_params = RNDNormIntReturnParams(
        1e-4, 0.0, 1.0, jnp.zeros((config["NUM_STEPS"],))
    )

    obs_norm_params = update_obs_norm_params(init_obs_norm_params, init_obs)
    target = TargetNetwork(256)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS_PER_DEVICE"])
    obsv, env_state = env.reset(reset_rng, env_params)

    # TRAIN LOOP
    def _update_step(runner_state, unused):
        # COLLECT TRAJECTORIES
        def _env_step(runner_state, unused):
            (
                train_state,
                pred_state,
                target_params,
                env_state,
                last_obs,
                rnd_int_return_norm_params,
                obs_norm_params,
                rng,
            ) = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = train_state.apply_fn(train_state.params, last_obs)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS_PER_DEVICE"])
            obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)

            # NORM THE OBS
            rnd_obs = ((obsv - obs_norm_params.mean) / jnp.sqrt(obs_norm_params.var)).clip(-5, 5)

            # INT REWARD
            tar_feat = target.apply(target_params, rnd_obs)
            pred_feat = pred_state.apply_fn(pred_state.params, rnd_obs)
            int_reward = jnp.square(jnp.linalg.norm((pred_feat - tar_feat), axis=1)) / 2

            transition = RNDTransition(
                done, action, value, reward, int_reward, log_prob, last_obs, info
            )
            runner_state = (
                train_state,
                pred_state,
                target_params,
                env_state,
                obsv,
                rnd_int_return_norm_params,
                obs_norm_params,
                rng,
            )
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

        # CALCULATE ADVANTAGE AND NORMALISE INT REWARDS
        (
            train_state,
            pred_state,
            target_params,
            env_state,
            last_obs,
            rnd_int_return_norm_params,
            obs_norm_params,
            rng,
        ) = runner_state
        _, last_val = train_state.apply_fn(train_state.params, last_obs)

        def _calculate_gae(traj_batch, last_val, rnd_int_return_norm_params):

            norm_int_reward, rnd_int_return_norm_params, _ = rnd_normalise_int_rewards(
                traj_batch, rnd_int_return_norm_params, config["INT_GAMMA"], jnp.zeros((1, 1))
            )

            norm_traj_batch = RNDTransition(
                traj_batch.done,
                traj_batch.action,
                traj_batch.value,
                traj_batch.reward,
                norm_int_reward,
                traj_batch.log_prob,
                traj_batch.obs,
                traj_batch.info,
            )

            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward, int_reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                    transition.int_reward,
                )
                delta = (
                    (reward + (config["INT_LAMBDA"] * int_reward))
                    + config["GAMMA"] * next_value * (1 - done)
                    - value
                )
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                norm_traj_batch,
                reverse=True,
                unroll=16,
            )
            return (
                advantages,
                advantages + traj_batch.value,
                norm_int_reward,
                rnd_int_return_norm_params,
            )

        advantages, targets, norm_int_rewards, rnd_int_return_norm_params = _calculate_gae(
            traj_batch, last_val, rnd_int_return_norm_params
        )

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):

            (
                train_state,
                pred_state,
                traj_batch,
                advantages,
                targets,
                obs_norm_params,
                rng,
            ) = update_state
            rng, _mask_rng = jax.random.split(rng)
            rng, _rng = jax.random.split(rng)

            def _update_minbatch(network_states, batch_info):
                traj_batch, advantages, targets, rnd_obs = batch_info
                train_state, pred_state = network_states

                def _rnd_loss(pred_params, rnd_obs):
                    tar_feat = target.apply(target_params, rnd_obs)
                    pred_feat = pred_state.apply_fn(pred_params, rnd_obs)
                    loss = jnp.square(jnp.linalg.norm((pred_feat - tar_feat), axis=1)) / 2

                    mask = jax.random.uniform(_mask_rng, (loss.shape[0],))
                    mask = (mask < config["UPDATE_PROPORTION"]).astype(jnp.float32)
                    loss = loss * mask
                    return loss.sum() / jnp.max(jnp.array([mask.sum(), 1]))

                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    pi, value = train_state.apply_fn(params, traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config["CLIP_EPS"], config["CLIP_EPS"]
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    total_loss = (
                        loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                (loss, (vloss, aloss, entropy)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    train_state.params, traj_batch, advantages, targets
                )

                rnd_loss, rnd_grads = jax.value_and_grad(_rnd_loss)(pred_state.params, rnd_obs)
                (loss, vloss, aloss, entropy, rnd_loss, grads, rnd_grads) = jax.lax.pmean(
                    (loss, vloss, aloss, entropy, rnd_loss, grads, rnd_grads), axis_name="devices"
                )
                train_state = train_state.apply_gradients(grads=grads)
                pred_state = pred_state.apply_gradients(grads=rnd_grads)
                return (train_state, pred_state), (loss, (vloss, aloss, entropy, rnd_loss))

            # UPDATE OBS NORM PARAMETERS
            obs_norm_params = update_obs_norm_params(
                obs_norm_params, traj_batch.obs.reshape(-1, init_obs.shape[-1])
            )
            # GET RND OBS
            rnd_obs = (
                (traj_batch.obs - obs_norm_params.mean) / jnp.sqrt(obs_norm_params.var)
            ).clip(-5, 5)

            rng, _rng = jax.random.split(rng)
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            assert (
                batch_size == config["NUM_STEPS"] * config["NUM_ENVS_PER_DEVICE"]
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets, rnd_obs)
            batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                shuffled_batch,
            )
            (train_state, pred_state), total_loss = jax.lax.scan(
                _update_minbatch, (train_state, pred_state), minibatches
            )
            update_state = (
                train_state,
                pred_state,
                traj_batch,
                advantages,
                targets,
                obs_norm_params,
                rng,
            )
            return update_state, total_loss

        update_state = (
            train_state,
            pred_state,
            traj_batch,
            advantages,
            targets,
            obs_norm_params,
            rng,
        )
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )
        train_state = update_state[0]
        pred_state = update_state[1]
        obs_norm_params = update_state[-2]
        metric = traj_batch.info
        rng = update_state[-1]
        if config.get("DEBUG"):

            def callback(info):
                return_values = info["returned_episode_returns"][info["returned_episode"]]
                timesteps = (
                    info["timestep"][info["returned_episode"]] * config["NUM_ENVS_PER_DEVICE"]
                )
                for t in range(len(timesteps)):
                    print(f"global step={timesteps[t]}, episodic return={return_values[t]}")

            jax.debug.callback(callback, metric)

        runner_state = (
            train_state,
            pred_state,
            target_params,
            env_state,
            last_obs,
            rnd_int_return_norm_params,
            obs_norm_params,
            rng,
        )
        return runner_state, (metric, traj_batch.int_reward, norm_int_rewards, loss_info)

    rng, _rng = jax.random.split(rng)
    runner_state = (
        train_state,
        pred_state,
        target_params,
        env_state,
        obsv,
        rnd_int_return_norm_params,
        obs_norm_params,
        _rng,
    )
    runner_state, extra_info = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
    metric, int_rewards, norm_int_rewards, rl_total_loss = extra_info
    metric["returned_episode_returns"] = metric["returned_episode_returns"].mean(-1)
    metric["returned_episode_returns"] = metric["returned_episode_returns"].reshape(
        metric["returned_episode_returns"].shape[0], -1
    )

    return {
        # "runner_state": runner_state,
        "metrics": metric,
        # "loss_info": loss,
        # "rl_total_loss": loss["total_loss"],
        # "rl_value_loss": loss["value_loss"],
        # "rl_actor_loss": loss["actor_loss"],
        # "rl_entrophy_loss": loss["entropy"],
        # "int_reward": int_reward,
        # "norm_int_reward": norm_int_reward,
        # "pred_loss": loss["pred_loss"],
    }


env_name = "ant"
# lambda_values = jnp.array(
#     [0.001, 0.0001, 0.0003, 0.0005, 0.0008, 0.01, 0.1, 0.003, 0.005, 0.02, 0.03, 0.05]
# ).sort()
lambda_values = jnp.array([0.001, 0.0001]).sort()
y_values = {}
for lambda_value in lambda_values:
    y_values[
        float(lambda_value)
    ] = {}  # Use float(lambda_value) to ensure dictionary keys are serializable
    config["INT_LAMBDA"] = lambda_value
    for step_int in step_intervals:
        t = time.time()
        rng = jax.random.PRNGKey(config["SEED"])
        config["STEP_INTERVAL"] = step_int
        config, env, env_params = make_config_env(config, env_name)

        rng = jax.random.split(rng, config["NUM_SEEDS"])
        rng, train_state, pred_state, target_params, init_rnd_obs = jax.vmap(
            make_train, out_axes=(1, 0, 0, 0, 0)
        )(rng)
        train_state = replicate(train_state, jax.local_devices())
        pred_state = replicate(pred_state, jax.local_devices())
        target_params = replicate(target_params, jax.local_devices())
        init_rnd_obs = replicate(init_rnd_obs, jax.local_devices())
        train_fn = jax.vmap(train, in_axes=(0, 0, 0, 0, 0))
        train_fn = jax.pmap(train_fn, axis_name="devices")
        output = jax.block_until_ready(
            train_fn(rng, train_state, pred_state, target_params, init_rnd_obs)
        )
        elapsed_time = time.time() - t
        print(output["metrics"]["returned_episode_returns"].shape)
        epi_ret = (
            (output["metrics"]["returned_episode_returns"].mean(0))
            .reshape(output["metrics"]["returned_episode_returns"].shape[1], -1)
            .T[-1]
        )
        del (
            output,
            train_state,
            pred_state,
        )
        samples = []
        for _ in range(3):
            key, resample_key = jax.random.split(key)
            samples.append(jax.random.choice(resample_key, epi_ret, shape=(10,), replace=True))

        epi_ret = np.array(samples).flatten()
        # Clear JAX caches
        jax.clear_caches()

        # Force Python garbage collection
        gc.collect()

        print(f"Memory cleared after processing {env_name}_{step_int}")

        print((time.time() - t) / 60)
        # Assuming `output` is your array

        # Use the last element of each row from 'epi_ret' as y-values
        y_values[float(lambda_value)][step_int] = epi_ret
        print(epi_ret.shape)


# Metric names corresponding to the data stored in y_values
metric_names = [
    "Episode Returns",
]


def normalize_curious_agent_returns(
    baseline_path, random_agent_path, curious_agent_last_episode_return
):

    # Load and clean random agent data
    random_agent_returns = np.load(random_agent_path)
    random_agent_returns = random_agent_returns[~np.isnan(random_agent_returns)]
    random_agent_returns = random_agent_returns[~np.isinf(random_agent_returns)]
    random_agent_mean = random_agent_returns.mean()
    print(f"Length of random agent array: {len(random_agent_returns)}")

    # Load and clean baseline data
    baseline_returns = np.load(baseline_path)
    baseline_last_episode_return = baseline_returns[-1]
    baseline_last_episode_return = baseline_last_episode_return[
        ~np.isnan(baseline_last_episode_return)
    ]
    baseline_last_episode_return = baseline_last_episode_return[
        ~np.isinf(baseline_last_episode_return)
    ]
    baseline_mean = baseline_last_episode_return.mean()
    print(f"Length of baseline array: {len(baseline_returns)}")

    # Clean curious agent data
    curious_agent_cleaned = curious_agent_last_episode_return[
        ~np.isnan(curious_agent_last_episode_return)
    ]
    curious_agent_cleaned = curious_agent_cleaned[~np.isinf(curious_agent_cleaned)]
    print(f"Length of curious agent array: {len(curious_agent_cleaned)}")

    # Check if baseline mean is less than random agent mean
    print(
        f"Is baseline mean ({baseline_mean}) less than random agent mean ({random_agent_mean})? {baseline_mean < random_agent_mean}"
    )

    # Normalize the curious agent returns between 0 and 1
    normalized_curious_agent_returns = (curious_agent_cleaned - random_agent_mean) / (
        baseline_mean - random_agent_mean
    )

    return normalized_curious_agent_returns

    # Construct the save path using the curious algorithm type
    # os.makedirs(save_path, exist_ok=True)

    # Save the normalized curious agent returns
    # normalized_curious_agent_file = os.path.join(save_path, "normalized_curious_agent_returns.npy")
    # np.save(normalized_curious_agent_file, normalized_curious_agent_returns)

    # Print the size of the saved file in MB
    # print(
    #     f"Size of normalized_curious_agent_returns.npy: {os.path.getsize(normalized_curious_agent_file) / (1024 * 1024):.2f} MB"
    # )

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Time taken to run the code: {elapsed_time:.2f} seconds")

    # return normalized_curious_agent_returns


# Initialize plotting
# for env_name in environments:
#     num_metrics = len(metric_names)
#     fig, axs = plt.subplots(num_metrics, 1, figsize=(12, 6 * num_metrics), sharex=False)
#     fig.suptitle(f"Training Metrics Over Time for {env_name}")

#     # Iterate over each metric
#     for idx, metric_name in enumerate(metric_names):
#         ax = axs[idx] if num_metrics > 1 else axs

#         # Plot each lambda's data for this metric
#         plotted = False
#         for lambda_value in lambda_values:
#             lambda_key = float(lambda_value)  # Ensure float key matches dictionary keys
#             if lambda_key in y_values and env_name in y_values[lambda_key]:
#                 metric_data = y_values[lambda_key][env_name][idx]

#                 # Convert JAX array to NumPy array to ensure it can be used with len()
#                 metric_data = np.array(metric_data)  # Ensure it's a NumPy array
#                 if len(metric_data) > 0:
#                     x_axis = range(1, len(metric_data) + 1)
#                     ax.plot(x_axis, metric_data, label=f"Lambda={lambda_value:.5f}")
#                     plotted = True

#         # Only add a legend if data was actually plotted
#         if plotted:
#             ax.set_title(metric_name)
#             ax.set_xlabel("Training Steps")
#             ax.set_ylabel(metric_name)
#             ax.legend()
#         else:
#             ax.set_title(f"{metric_name} (no data)")
#             ax.set_xlabel("Training Steps")
#             ax.set_ylabel(metric_name)

#     # Adjust layout and save the figure
#     plt.subplots_adjust(hspace=0.4)  # Adjust vertical spacing between plots
#     plt.savefig(
#         f"MetaLearnCuriosity/hyperparameter_sweep/MiniGrid/{env_name}_metrics_over_time_byol.png"
#     )
#     plt.close(fig)
#     gc.collect()

# # Scatter plot for final episode returns vs. lambda values
# for env_name in environments:
#     lambda_values = []
#     final_returns = []

#     # Collect data for plotting
#     for lambda_value, env_data in y_values.items():
#         epi_ret = env_data[env_name][0]  # index 0 for episode returns
#         epi_ret = np.array(env_data[env_name][0])
#         if epi_ret.size > 0:  # Ensure there is at least one return value
#             final_returns.append(epi_ret)
#             lambda_values.append(lambda_value)

#     # Sort lambda values for consistent plotting
#     sorted_indices = sorted(range(len(lambda_values)), key=lambda k: lambda_values[k])
#     sorted_lambda_values = [lambda_values[i] for i in sorted_indices]
#     sorted_final_returns = [final_returns[i] for i in sorted_indices]

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.scatter(range(1, len(sorted_lambda_values) + 1), sorted_final_returns, color="blue")
#     plt.title(f"Final Episode Returns vs. Lambda for {env_name}")
#     plt.xlabel("Lambda Value (Indexed)")
#     plt.ylabel("Final Episode Return")
#     plt.xticks(
#         range(1, len(sorted_lambda_values) + 1),
#         [f"{lv:.5f}" for lv in sorted_lambda_values],
#         rotation=45,
#     )
#     plt.grid(True)

#     # Save and close plot
#     plt.savefig(
#         f"MetaLearnCuriosity/hyperparameter_sweep/Brax/{env_name}_final_episode_returns_vs_lambda_RND_{config['STEP_INTERVAL']}.png"
#     )
#     plt.close()  # Close the plot to free up memory
#     gc.collect()
# Scatter plot for final episode returns vs. lambda values


# save_dir = f"/home/batsy/MetaLearnCuriosity/MetaLearnCuriosity/hyperparameter_sweep/Brax_Arrays/Brax_{config['STEP_INTERVAL']}/RND"
# os.makedirs(save_dir, exist_ok=True)

# # First loop remains the same
# for lambda_value, env_data in y_values.items():
#     for env_name, epi_ret in env_data.items():
#         y_values[lambda_value][env_name] = normalize_curious_agent_returns(
#             f"/home/batsy/MetaLearnCuriosity/MetaLearnCuriosity/experiments/delayed_brax_baseline_ppo_{config['STEP_INTERVAL']}/{env_name}/metric_seeds_episode_return.npy",
#             f"/home/batsy/MetaLearnCuriosity/MetaLearnCuriosity/experiments/random_agents/{env_name}_epi_rets.npy",
#             epi_ret.T,
#         )

# # Second loop: save normalized returns for each lambda
# for lambda_value, env_data in y_values.items():
#     normalized_returns = []
#     for env_name, normalized_epi_ret in env_data.items():
#         normalized_returns.append(normalized_epi_ret)

#     # Save the normalized returns array
#     save_path = os.path.join(save_dir, f"{round(lambda_value, 4):.4e}_all_normalised_returns.npy")
#     np.save(save_path, np.array(normalized_returns))
#     y_values[lambda_value] = np.array(normalized_returns)

# # Now load the saved arrays and calculate statistics
# lambda_values = sorted(y_values.keys())
# means = []
# ci_lows = []
# ci_highs = []

# for lambda_value in lambda_values:
#     # Load the saved normalized returns
#     load_path = os.path.join(save_dir, f"{round(lambda_value, 4):.4e}_all_normalised_returns.npy")
#     normalized_returns = np.load(load_path).flatten()

#     # Calculate statistics
#     mean_value = np.mean(normalized_returns)
#     means.append(mean_value)

#     ci = bootstrap(
#         (normalized_returns,),
#         np.mean,
#         confidence_level=0.95,
#         method="percentile",
#     )
#     ci_lows.append(ci.confidence_interval.low)
#     ci_highs.append(ci.confidence_interval.high)

# # Create evenly spaced x positions
# x_positions = np.arange(len(lambda_values))

# plt.figure(figsize=(15, 8))  # Maintain larger figure size for clarity

# # Create the bar plot
# plt.bar(
#     x_positions,
#     means,
#     yerr=[np.array(means) - np.array(ci_lows), np.array(ci_highs) - np.array(means)],
#     capsize=5,
#     color="skyblue",
#     edgecolor="black",
#     width=0.6,
# )

# # Add grid
# plt.grid(True)

# # Set x-axis ticks and labels
# plt.xticks(x_positions, [f"{lv:.4f}" for lv in lambda_values], rotation=45, ha="right")

# # Adjust y-axis to show both positive and negative values
# y_min = min(min(ci_lows), 0)
# y_max = max(ci_highs)
# plt.ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))

# # Add a horizontal line at y=0
# plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

# # Label the axes
# plt.xlabel("Lambda Value", fontsize=12)
# plt.ylabel("Mean Normalized Episode Return", fontsize=12)
# plt.title("Mean Normalized Episode Return vs Lambda Value", fontsize=14)

# # Adjust layout
# plt.tight_layout()

# # Save the plot
# plt.savefig(
#     f"/home/batsy/MetaLearnCuriosity/MetaLearnCuriosity/hyperparameter_sweep/Brax/Brax_normalized_returns_histogram_RND_{config['STEP_INTERVAL']}.png",
#     dpi=300,
# )
# plt.close()


# Update save directory structure for fixed environment and varying step intervals
# 1. First save the data
save_dir = f"/home/batsy/MetaLearnCuriosity/MetaLearnCuriosity/hyperparameter_sweep/Brax_Arrays/Brax_{env_name}/RND"
os.makedirs(save_dir, exist_ok=True)

# Save data for each lambda and step interval
for lambda_value, step_data in y_values.items():
    for step_int, returns in step_data.items():
        save_path = os.path.join(save_dir, f"lambda_{lambda_value:.6f}_step_{step_int}_returns.npy")
        np.save(save_path, returns)

# 2. Create individual plots for each step interval
for step_int in step_intervals:
    means = []
    ci_lows = []
    ci_highs = []
    lambda_vals = []

    for lambda_value in sorted(y_values.keys()):
        returns = y_values[lambda_value][step_int]
        mean_return = np.mean(returns)
        means.append(mean_return)
        lambda_vals.append(lambda_value)

        # Calculate confidence intervals
        ci = bootstrap(
            (returns,), np.mean, confidence_level=0.95, method="percentile", n_resamples=10000
        )
        ci_lows.append(ci.confidence_interval.low)
        ci_highs.append(ci.confidence_interval.high)

    # Create bar plot
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(lambda_vals))
    plt.bar(
        x_pos,
        means,
        yerr=[np.array(means) - np.array(ci_lows), np.array(ci_highs) - np.array(means)],
        capsize=5,
        color="skyblue",
        edgecolor="black",
    )

    plt.title(f"{env_name} Environment - Step Interval {step_int}")
    plt.xlabel("Lambda Values")
    plt.ylabel("Mean Episode Return")
    plt.xticks(x_pos, [f"{lv:.6f}" for lv in lambda_vals], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(save_dir, f"returns_step_{step_int}.png"))
    plt.close()

# 3. Create aggregate plot across all step intervals
aggregate_means = []
aggregate_ci_lows = []
aggregate_ci_highs = []
lambda_vals = sorted(y_values.keys())

for lambda_value in lambda_vals:
    # Collect all returns for this lambda across all step intervals
    all_returns = []
    for step_int in step_intervals:
        all_returns.extend(y_values[lambda_value][step_int])

    # Convert to numpy array
    all_returns = np.array(all_returns)

    # Calculate statistics
    mean_return = np.mean(all_returns)
    aggregate_means.append(mean_return)

    # Calculate confidence intervals
    ci = bootstrap(
        (all_returns,), np.mean, confidence_level=0.95, method="percentile", n_resamples=10000
    )
    aggregate_ci_lows.append(ci.confidence_interval.low)
    aggregate_ci_highs.append(ci.confidence_interval.high)

# Create aggregate bar plot
plt.figure(figsize=(12, 6))
x_pos = np.arange(len(lambda_vals))
plt.bar(
    x_pos,
    aggregate_means,
    yerr=[
        np.array(aggregate_means) - np.array(aggregate_ci_lows),
        np.array(aggregate_ci_highs) - np.array(aggregate_means),
    ],
    capsize=5,
    color="lightgreen",
    edgecolor="black",
)

plt.title(f"{env_name} Environment - Aggregate Across All Step Intervals")
plt.xlabel("Lambda Values")
plt.ylabel("Mean Episode Return")
plt.xticks(x_pos, [f"{lv:.6f}" for lv in lambda_vals], rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save aggregate plot
plt.savefig(os.path.join(save_dir, "aggregate_returns_all_steps.png"))
plt.close()