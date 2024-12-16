# Taken from:
# https://github.com/corl-team/xland-minigrid/blob/main/training/train_single_task.py

import gc
import os
import shutil
import time

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState
from scipy.stats import bootstrap

import wandb
from MetaLearnCuriosity.agents.nn import (
    MiniGridActorCriticRNN,
    PredictorNetwork,
    TargetNetwork,
)
from MetaLearnCuriosity.checkpoints import Save
from MetaLearnCuriosity.logger import WBLogger
from MetaLearnCuriosity.utils import (
    ObsNormParams,
    RNDMiniGridTransition,
    RNDNormIntReturnParams,
    compress_output_for_reasoning,
    make_obs_gymnax_discrete,
    process_output_general,
    rnd_calculate_gae,
    rnd_minigrid_ppo_update_networks,
    update_obs_norm_params,
)
from MetaLearnCuriosity.wrappers import (
    FlattenObservationWrapper,
    LogWrapper,
    MiniGridGymnax,
    VecEnv,
)

jax.config.update("jax_threefry_partitionable", True)

environments = [
    "MiniGrid-DoorKey-16x16",
    "MiniGrid-DoorKey-8x8",
    "MiniGrid-DoorKey-6x6",
    "MiniGrid-DoorKey-5x5",
]

config = {
    "NUM_SEEDS": 10,
    "PROJECT": "MetaLearnCuriosity",
    "RUN_NAME": "rnd_minigrid",
    "BENCHMARK_ID": None,
    "RULESET_ID": None,
    "USE_CNNS": False,
    # Agent
    "ACTION_EMB_DIM": 16,
    "RNN_HIDDEN_DIM": 1024,
    "RNN_NUM_LAYERS": 1,
    "HEAD_HIDDEN_DIM": 256,
    # Training
    "NUM_ENVS": 8192,
    "NUM_STEPS": 16,
    "UPDATE_EPOCHS": 1,
    "NUM_MINIBATCHES": 16,
    "TOTAL_TIMESTEPS": 50_000_000,
    "LR": 0.001,
    "PRED_LR": 1e-3,
    "CLIP_EPS": 0.2,
    "GAMMA": 0.99,
    "INT_GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "INT_LAMBDA": 0.01,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "EVAL_EPISODES": 80,
    "SEED": 98,
}

key = jax.random.PRNGKey(97776)


def make_env_config(config, env_name):
    num_devices = jax.local_device_count()
    config["ENV_NAME"] = env_name
    assert config["NUM_ENVS"] % num_devices == 0
    env = MiniGridGymnax(config["ENV_NAME"])
    env_params = env._env_params
    env_eval = MiniGridGymnax(config["ENV_NAME"])
    if config["USE_CNNS"]:
        observations_shape = env.observation_space(env_params).shape[0]
    else:
        env = FlattenObservationWrapper(env)
        observations_shape = env.observation_space(env_params).shape
    print(observations_shape)
    env_eval = LogWrapper(env_eval)
    env = LogWrapper(env)
    env = VecEnv(env)
    num_devices = jax.local_device_count()
    config["NUM_ENVS_PER_DEVICE"] = config["NUM_ENVS"] // num_devices
    config["TOTAL_TIMESTEPS_PER_DEVICE"] = config["TOTAL_TIMESTEPS"] // num_devices
    config["EVAL_EPISODES_PER_DEVICE"] = config["EVAL_EPISODES"] // num_devices
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS_PER_DEVICE"] // config["NUM_STEPS"] // config["NUM_ENVS_PER_DEVICE"]
    )
    assert config["NUM_ENVS_PER_DEVICE"] >= 4
    config["UPDATE_PROPORTION"] = 4 / config["NUM_ENVS_PER_DEVICE"]
    print(f"Num devices: {num_devices}, Num updates: {config['NUM_UPDATES']}")
    return observations_shape, config, env, env_params


def make_train(rng):
    rng, _rng = jax.random.split(rng)
    rng, _pred_rng = jax.random.split(rng)
    rng, _tar_rng = jax.random.split(rng)
    rng, _init_obs_rng = jax.random.split(rng)

    num_actions = env.action_space(env_params).n

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

    network = MiniGridActorCriticRNN(
        num_actions=num_actions,
        action_emb_dim=config["ACTION_EMB_DIM"],
        rnn_hidden_dim=config["RNN_HIDDEN_DIM"],
        rnn_num_layers=config["RNN_NUM_LAYERS"],
        head_hidden_dim=config["HEAD_HIDDEN_DIM"],
        use_cnns=config["USE_CNNS"],
    )
    target = TargetNetwork(256)
    predictor = PredictorNetwork(256)

    init_obs = {
        "observation": jnp.zeros((config["NUM_ENVS_PER_DEVICE"], 1, *observations_shape)),
        "prev_action": jnp.zeros((config["NUM_ENVS_PER_DEVICE"], 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((config["NUM_ENVS_PER_DEVICE"], 1)),
    }
    init_hstate = network.initialize_carry(batch_size=config["NUM_ENVS_PER_DEVICE"])
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params = network.init(_rng, init_obs, init_hstate)
    target_params = target.init(_tar_rng, init_x)
    pred_params = predictor.init(_pred_rng, init_x)

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.inject_hyperparams(optax.adam)(learning_rate=linear_schedule, eps=1e-8),  # eps=1e-5
    )
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
    pred_tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["PRED_LR"], eps=1e-5),
    )

    pred_state = TrainState.create(apply_fn=predictor.apply, params=pred_params, tx=pred_tx)

    rng = jax.random.split(rng, jax.local_device_count())

    return init_hstate, train_state, pred_state, target_params, rng, _init_obs_rng


def train(rng, init_hstate, train_state, pred_state, target_params, init_obs_rng):

    # INIT OBS NORM PARAMS:
    random_rollout = make_obs_gymnax_discrete(config["NUM_ENVS_PER_DEVICE"], env, env_params, 1)
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
    rng, _rng = jax.random.split(rng)

    reset_rng = jax.random.split(_rng, config["NUM_ENVS_PER_DEVICE"])

    obsv, env_state = env.reset(reset_rng, env_params)
    prev_action = jnp.zeros(config["NUM_ENVS_PER_DEVICE"], dtype=jnp.int32)
    prev_reward = jnp.zeros(config["NUM_ENVS_PER_DEVICE"])

    # TRAIN LOOP

    def _update_step(runner_state, _):

        # COLLECT TRAJECTORIES

        def _env_step(runner_state, _):
            (
                rng,
                train_state,
                pred_state,
                target_params,
                obs_norm_params,
                rnd_int_return_norm_params,
                env_state,
                prev_obs,
                prev_action,
                prev_reward,
                prev_hstate,
            ) = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            dist, value, hstate = train_state.apply_fn(
                train_state.params,
                {
                    # [batch_size, seq_len=1, ...]
                    "observation": prev_obs[:, None],
                    "prev_action": prev_action[:, None],
                    "prev_reward": prev_reward[:, None],
                },
                prev_hstate,
            )
            action, log_prob = dist.sample_and_log_prob(seed=_rng)
            # squeeze seq_len where possible
            action, value, log_prob = action.squeeze(1), value.squeeze(1), log_prob.squeeze(1)

            # STEP ENV
            rng_step = jax.random.split(rng, config["NUM_ENVS_PER_DEVICE"])
            obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
            # NORM THE OBS
            rnd_obs = ((obsv - obs_norm_params.mean) / jnp.sqrt(obs_norm_params.var)).clip(-5, 5)

            # INT REWARD
            tar_feat = target.apply(target_params, rnd_obs)
            pred_feat = pred_state.apply_fn(pred_state.params, rnd_obs)
            int_reward = jnp.square(jnp.linalg.norm((pred_feat - tar_feat), axis=1)) / 2
            transition = RNDMiniGridTransition(
                done=done,
                action=action,
                value=value,
                reward=reward,
                int_reward=int_reward,
                log_prob=log_prob,
                obs=prev_obs,
                prev_action=prev_action,
                prev_reward=prev_reward,
                info=info,
            )
            runner_state = (
                rng,
                train_state,
                pred_state,
                target_params,
                obs_norm_params,
                rnd_int_return_norm_params,
                env_state,
                obsv,
                action,
                reward,
                hstate,
            )
            return runner_state, transition

        initial_hstate = runner_state[-1]
        runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

        # CALCULATE ADVANTAGE

        (
            rng,
            train_state,
            pred_state,
            target_params,
            obs_norm_params,
            rnd_int_return_norm_params,
            env_state,
            prev_obs,
            prev_action,
            prev_reward,
            hstate,
        ) = runner_state

        _, last_val, _ = train_state.apply_fn(
            train_state.params,
            {
                "observation": prev_obs[:, None],
                "prev_action": prev_action[:, None],
                "prev_reward": prev_reward[:, None],
            },
            hstate,
        )

        advantages, targets, rnd_int_return_norm_params, norm_int_rewards = rnd_calculate_gae(
            transitions,
            last_val.squeeze(1),
            config["GAMMA"],
            config["INT_GAMMA"],
            config["GAE_LAMBDA"],
            config["INT_LAMBDA"],
            rnd_int_return_norm_params,
        )

        # UPDATE NETWORK
        def _update_epoch(update_state, _):
            def _update_minbatch(train_states, batch_info):
                init_hstate, transitions, advantages, targets, rnd_obs = batch_info
                train_state, pred_state = train_states
                (new_train_state, pred_state), update_info = rnd_minigrid_ppo_update_networks(
                    train_state=train_state,
                    pred_state=pred_state,
                    target_params=target_params,
                    _mask_rng=_mask_rng,
                    transitions=transitions,
                    rnd_obs=rnd_obs,
                    init_hstate=init_hstate.squeeze(1),
                    advantages=advantages,
                    targets=targets,
                    update_prop=config["UPDATE_PROPORTION"],
                    clip_eps=config["CLIP_EPS"],
                    vf_coef=config["VF_COEF"],
                    ent_coef=config["ENT_COEF"],
                )
                return (new_train_state, pred_state), update_info

            (
                rng,
                train_state,
                pred_state,
                obs_norm_params,
                init_hstate,
                transitions,
                advantages,
                targets,
            ) = update_state

            # MINIBATCHES PREPARATION
            # UPDATE OBS NORM PARAMETERS
            obs_norm_params = update_obs_norm_params(
                obs_norm_params, transitions.obs.reshape(-1, init_obs.shape[-1])
            )
            # GET RND OBS
            rnd_obs = (
                (transitions.obs - obs_norm_params.mean) / jnp.sqrt(obs_norm_params.var)
            ).clip(-5, 5)

            rng, _rng = jax.random.split(rng)
            rng, _mask_rng = jax.random.split(rng)

            permutation = jax.random.permutation(_rng, config["NUM_ENVS_PER_DEVICE"])
            # [seq_len, batch_size, ...]
            batch = (init_hstate, transitions, advantages, targets, rnd_obs)
            # [batch_size, seq_len, ...], as our model assumes
            batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

            shuffled_batch = jtu.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
            # [num_minibatches, minibatch_size, ...]
            minibatches = jtu.tree_map(
                lambda x: jnp.reshape(x, (config["NUM_MINIBATCHES"], -1) + x.shape[1:]),
                shuffled_batch,
            )
            (train_state, pred_state), update_info = jax.lax.scan(
                _update_minbatch, (train_state, pred_state), minibatches
            )

            update_state = (
                rng,
                train_state,
                pred_state,
                obs_norm_params,
                init_hstate,
                transitions,
                advantages,
                targets,
            )
            return update_state, update_info

        # [seq_len, batch_size, num_layers, hidden_dim]
        init_hstate = initial_hstate[None, :]
        update_state = (
            rng,
            train_state,
            pred_state,
            obs_norm_params,
            init_hstate,
            transitions,
            advantages,
            targets,
        )
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )

        # averaging over minibatches then over epochs
        # loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)

        # EVALUATE AGENT
        # rng, _rng = jax.random.split(rng)
        # eval_rng = jax.random.split(_rng, num=config["EVAL_EPISODES_PER_DEVICE"])

        # # vmap only on rngs
        # eval_stats = jax.vmap(rnn_rollout, in_axes=(0, None, None, None, None, None))(
        #     eval_rng,
        #     env_eval,
        #     env_params,
        #     train_state,
        #     # TODO: make this as a static method mb?
        #     jnp.zeros((1, config["RNN_NUM_LAYERS"], config["RNN_HIDDEN_DIM"])),
        #     1,
        # )
        # eval_stats = jax.lax.pmean(eval_stats, axis_name="devices")
        # loss_info.update(
        #     {
        #         "eval/returns": eval_stats.reward.mean(0),
        #         "eval/lengths": eval_stats.length.mean(0),
        #         "lr": train_state.opt_state[-1].hyperparams["learning_rate"],
        #     }
        # )

        rng, train_state, pred_state, obs_norm_params = update_state[:4]
        traj_batch = update_state[5]
        metric = traj_batch.info
        runner_state = (
            rng,
            train_state,
            pred_state,
            target_params,
            obs_norm_params,
            rnd_int_return_norm_params,
            env_state,
            prev_obs,
            prev_action,
            prev_reward,
            hstate,
        )
        return runner_state, (metric, loss_info, traj_batch.int_reward, norm_int_rewards)

    runner_state = (
        rng,
        train_state,
        pred_state,
        target_params,
        obs_norm_params,
        rnd_int_return_norm_params,
        env_state,
        obsv,
        prev_action,
        prev_reward,
        init_hstate,
    )
    runner_state, loss_info = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
    metric, loss, int_reward, norm_int_reward = loss_info
    return {
        # "train_states": runner_state[1],
        "metrics": metric,
        # "loss_info": loss,
        # "norm_int_reward": norm_int_reward,
        # "int_reward": int_reward,
        # "rl_total_loss": loss["total_loss"],
        # "rl_value_loss": loss["value_loss"],
        # "rl_actor_loss": loss["actor_loss"],
        # "rl_entrophy_loss": loss["entropy"],
        # "rnd_loss": loss["rnd_loss"],
    }


lambda_values = jnp.array(
    [0.001, 0.0001, 0.0003, 0.0005, 0.0008, 0.01, 0.1, 0.003, 0.005, 0.02, 0.03, 0.05]
).sort()
# lambda_values = jnp.array([0.001, 0.0001]).sort()
y_values = {}
for lambda_value in lambda_values:
    y_values[
        float(lambda_value)
    ] = {}  # Use float(lambda_value) to ensure dictionary keys are serializable
    config["INT_LAMBDA"] = lambda_value
    for env_name in environments:

        observations_shape, config, env, env_params = make_env_config(config, env_name)

        # experiments
        rng = jax.random.PRNGKey(config["SEED"])
        rng = jax.random.split(rng, config["NUM_SEEDS"])
        (init_hstate, train_state, pred_state, target_params, rng, init_obs_rng) = jax.jit(
            jax.vmap(make_train, out_axes=(0, 0, 0, 0, 0, 0, 1))
        )(rng)
        init_hstate = replicate(init_hstate, jax.local_devices())
        train_state = replicate(train_state, jax.local_devices())
        pred_state = replicate(pred_state, jax.local_devices())
        target_params = replicate(target_params, jax.local_devices())
        init_obs_rng = replicate(init_obs_rng, jax.local_devices)
        # init_hstate, train_state, pred_state, target_params, rng, _init_obs_rng
        train_fn = jax.vmap(train)
        train_fn = jax.pmap(train_fn, axis_name="devices")
        print(f"Training in {config['ENV_NAME']}")
        t = time.time()
        output = jax.block_until_ready(
            train_fn(
                rng,
                init_hstate,
                train_state,
                pred_state,
                target_params,
                init_obs_rng,
            )
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
            init_hstate,
            train_state,
            pred_state,
            target_params,
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

        print(f"Memory cleared after processing {env_name}")

        print((time.time() - t) / 60)
        # Assuming `output` is your array

        # Use the last element of each row from 'epi_ret' as y-values
        y_values[float(lambda_value)][env_name] = epi_ret
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


# 1 save data
save_dir = f"/home/batsy/MetaLearnCuriosity/MetaLearnCuriosity/hyperparameter_sweep/MinGrid_Arrays/MinGrid_{env_name}/RND"
os.makedirs(save_dir, exist_ok=True)

# Save data for each lambda and step interval
for lambda_value, env_data in y_values.items():
    for env_name, returns in env_data.items():
        save_path = os.path.join(save_dir, f"lambda_{lambda_value:.6f}_{env_name}_returns.npy")
        np.save(save_path, returns)

# 2. Create individual plots for each step interval
for env_name in environments:
    means = []
    ci_lows = []
    ci_highs = []
    lambda_vals = []

    for lambda_value in sorted(y_values.keys()):
        returns = y_values[lambda_value][env_name]
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

    plt.title(f"{env_name} Environment")
    plt.xlabel("Lambda Values")
    plt.ylabel("Mean Episode Return")
    plt.xticks(x_pos, [f"{lv:.6f}" for lv in lambda_vals], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(save_dir, f"returns_{env_name}.png"))
    plt.close()

# 3. Create aggregate plot across all step intervals
aggregate_means = []
aggregate_ci_lows = []
aggregate_ci_highs = []
lambda_vals = sorted(y_values.keys())

for lambda_value in lambda_vals:
    # Collect all returns for this lambda across all step intervals
    all_returns = []
    for env_name in env_name:
        all_returns.extend(y_values[lambda_value][env_name])

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

plt.title(f"{env_name} Environment - Aggregate Across All Step Sizes")
plt.xlabel("Lambda Values")
plt.ylabel("Mean Episode Return")
plt.xticks(x_pos, [f"{lv:.6f}" for lv in lambda_vals], rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save aggregate plot
plt.savefig(os.path.join(save_dir, "aggregate_returns_all_steps.png"))
plt.close()
