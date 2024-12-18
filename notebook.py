# Taken from:
# https://github.com/corl-team/xland-minigrid/blob/main/training/train_single_task.py

import gc
import os
import time

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState
from scipy.stats import bootstrap

from MetaLearnCuriosity.agents.nn import (
    BYOLTarget,
    CloseScannedRNN,
    MiniGridActorCriticRNN,
    MiniGridBYOLPredictor,
    OpenScannedRNN,
)
from MetaLearnCuriosity.checkpoints import Save
from MetaLearnCuriosity.logger import WBLogger
from MetaLearnCuriosity.utils import BYOLMiniGridTransition as Transition
from MetaLearnCuriosity.utils import (
    BYOLRewardNorm,
    byol_calculate_gae,
    byol_minigrid_ppo_update_networks,
    rnn_rollout,
)
from MetaLearnCuriosity.wrappers import (
    FlattenObservationWrapper,
    LogWrapper,
    MiniGridGymnax,
    VecEnv,
)

jax.config.update("jax_threefry_partitionable", True)
key = jax.random.PRNGKey(76)
environments = [
    "MiniGrid-DoorKey-8x8",
    "MiniGrid-Empty-16x16",
    "MiniGrid-EmptyRandom-16x16",
    "MiniGrid-FourRooms",
    "MiniGrid-MemoryS16",
    "MiniGrid-Unlock",
]

config = {
    "NUM_SEEDS": 10,
    "PROJECT": "MetaLearnCuriosity",
    "RUN_NAME": "minigrid-ppo-baseline",
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
    "CLIP_EPS": 0.2,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "EVAL_EPISODES": 80,
    "SEED": 42,
    "ANNEAL_PRED_LR": False,
    "DEBUG": False,
    "PRED_LR": 0.001,
    "REW_NORM_PARAMETER": 0.99,
    "EMA_PARAMETER": 0.99,
}


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
    env_eval = LogWrapper(env_eval)
    env = LogWrapper(env)
    num_devices = jax.local_device_count()
    config["NUM_ENVS_PER_DEVICE"] = config["NUM_ENVS"] // num_devices
    config["TOTAL_TIMESTEPS_PER_DEVICE"] = config["TOTAL_TIMESTEPS"] // num_devices
    config["EVAL_EPISODES_PER_DEVICE"] = config["EVAL_EPISODES"] // num_devices
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS_PER_DEVICE"] // config["NUM_STEPS"] // config["NUM_ENVS_PER_DEVICE"]
    )
    print(f"Num devices: {num_devices}, Num updates: {config['NUM_UPDATES']}")
    return observations_shape, config, env, env_params


def make_train(rng):
    rng, _rng = jax.random.split(rng)
    rng, _tar_rng = jax.random.split(rng)
    # rng, _en_rng = jax.random.split(rng)
    rng, _pred_rng = jax.random.split(rng)
    num_actions = env.action_space(env_params).n

    def pred_linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["PRED_LR"] * frac

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac
        # INIT network

    target = BYOLTarget(256)
    pred = MiniGridBYOLPredictor(256, num_actions)
    network = MiniGridActorCriticRNN(
        num_actions=num_actions,
        action_emb_dim=config["ACTION_EMB_DIM"],
        rnn_hidden_dim=config["RNN_HIDDEN_DIM"],
        rnn_num_layers=config["RNN_NUM_LAYERS"],
        head_hidden_dim=config["HEAD_HIDDEN_DIM"],
        use_cnns=config["USE_CNNS"],
    )
    init_obs = {
        "observation": jnp.zeros((config["NUM_ENVS_PER_DEVICE"], 1, *observations_shape)),
        "prev_action": jnp.zeros((config["NUM_ENVS_PER_DEVICE"], 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((config["NUM_ENVS_PER_DEVICE"], 1)),
    }
    init_hstate = network.initialize_carry(batch_size=config["NUM_ENVS_PER_DEVICE"])
    init_x = jnp.zeros((config["NUM_ENVS_PER_DEVICE"], 1, *observations_shape))
    init_action = jnp.zeros((config["NUM_ENVS_PER_DEVICE"], 1), dtype=jnp.int32)
    close_init_hstate = pred.initialize_carry(config["NUM_ENVS_PER_DEVICE"])
    open_init_hstate = pred.initialize_carry(config["NUM_ENVS_PER_DEVICE"])
    print(init_hstate.shape, close_init_hstate.shape, open_init_hstate.shape)
    init_bt = jnp.zeros((config["NUM_ENVS_PER_DEVICE"], 1, 256))
    init_pred_input = (init_bt, init_x, init_action)
    pred_params = pred.init(_pred_rng, close_init_hstate, open_init_hstate, init_pred_input)
    target_params = target.init(_tar_rng, init_x)

    network_params = network.init(_rng, init_obs, init_hstate)
    pred_params = pred.init(_pred_rng, close_init_hstate, open_init_hstate, init_pred_input)
    target_params = target.init(_tar_rng, init_x)

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.inject_hyperparams(optax.adam)(learning_rate=linear_schedule, eps=1e-8),  # eps=1e-5
    )
    if config["ANNEAL_PRED_LR"]:
        pred_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=pred_linear_schedule, eps=1e-5),
        )
    else:
        pred_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["PRED_LR"], eps=1e-5),
        )
    pred_state = TrainState.create(
        apply_fn=pred.apply,
        params=pred_params,
        tx=pred_tx,
    )

    target_state = TrainState.create(
        apply_fn=target.apply,
        params=target_params,
        tx=pred_tx,
    )
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
    # env = VecEnv(env)
    rng = jax.random.split(rng, jax.local_device_count())

    return (
        init_hstate,
        close_init_hstate,
        open_init_hstate,
        train_state,
        pred_state,
        target_state,
        rng,
    )


def train(
    rng, init_hstate, close_init_hstate, open_init_hstate, train_state, pred_state, target_state
):
    rng, _rng = jax.random.split(rng)

    reset_rng = jax.random.split(_rng, config["NUM_ENVS_PER_DEVICE"])
    # INIT STUFF FOR OPTIMIZATION AND NORMALIZATION
    update_target_counter = 0
    byol_reward_norm_params = BYOLRewardNorm(0, 0, 1, 0)

    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
    prev_action = jnp.zeros(config["NUM_ENVS_PER_DEVICE"], dtype=jnp.int32)
    prev_reward = jnp.zeros(config["NUM_ENVS_PER_DEVICE"])
    prev_bt = jnp.zeros((config["NUM_ENVS_PER_DEVICE"], 1, 256))

    # TRAIN LOOP

    def _update_step(runner_state, _):

        # COLLECT TRAJECTORIES

        def _env_step(runner_state, _):
            (
                rng,
                train_state,
                pred_state,
                target_state,
                env_state,
                prev_obs,
                prev_action,
                prev_reward,
                prev_bt,
                prev_hstate,
                close_prev_hstate,
                open_prev_hstate,
                byol_reward_norm_params,
                update_target_counter,
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
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                rng_step, env_state, action, env_params
            )
            # INT REWARD
            tar_obs = target_state.apply_fn(target_state.params, obsv[:, None])
            pred_input = (prev_bt, prev_obs[:, None], prev_action[:, None])
            pred_obs, new_bt, new_close_hstate, new_open_hstate = pred_state.apply_fn(
                pred_state.params, close_prev_hstate, open_prev_hstate, pred_input
            )
            pred_norm = (pred_obs.squeeze(1)) / (
                jnp.linalg.norm(pred_obs.squeeze(1), axis=-1, keepdims=True)
            )
            tar_norm = jax.lax.stop_gradient(
                (tar_obs.squeeze(1)) / (jnp.linalg.norm(tar_obs.squeeze(1), axis=-1, keepdims=True))
            )

            int_reward = jnp.square(jnp.linalg.norm((pred_norm - tar_norm), axis=-1)) * (1 - done)
            transition = Transition(
                done=done,
                action=action,
                value=value,
                reward=reward,
                int_reward=int_reward,
                log_prob=log_prob,
                obs=prev_obs,
                next_obs=obsv,
                prev_action=prev_action,
                prev_reward=prev_reward,
                prev_bt=prev_bt,
                info=info,
            )
            runner_state = (
                rng,
                train_state,
                pred_state,
                target_state,
                env_state,
                obsv,
                action,
                reward,
                new_bt,
                hstate,
                new_close_hstate,
                new_open_hstate,
                byol_reward_norm_params,
                update_target_counter,
            )
            return runner_state, transition

        initial_hstate, close_initial_hstate, open_initial_hstate = runner_state[9:12]
        runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

        # CALCULATE ADVANTAGE

        (
            rng,
            train_state,
            pred_state,
            target_state,
            env_state,
            prev_obs,
            prev_action,
            prev_reward,
            prev_bt,
            hstate,
            close_hstate,
            open_hstate,
            byol_reward_norm_params,
            update_target_counter,
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

        advantages, targets, norm_int_reward, byol_reward_norm_params = byol_calculate_gae(
            transitions,
            last_val.squeeze(1),
            config["GAMMA"],
            config["GAE_LAMBDA"],
            config["INT_LAMBDA"],
            config["REW_NORM_PARAMETER"],
            byol_reward_norm_params,
        )

        # UPDATE NETWORK
        def _update_epoch(update_state, _):
            def _update_minbatch(train_states, batch_info):
                (
                    init_hstate,
                    close_init_hstate,
                    open_init_hstate,
                    transitions,
                    advantages,
                    targets,
                ) = batch_info
                train_state, pred_state, target_state, update_target_counter = train_states
                (
                    new_train_state,
                    pred_state,
                    target_state,
                    update_target_counter,
                ), update_info = byol_minigrid_ppo_update_networks(
                    train_state=train_state,
                    pred_state=pred_state,
                    target_state=target_state,
                    transitions=transitions,
                    init_hstate=init_hstate.squeeze(1),
                    init_close_hstate=close_init_hstate.squeeze(1),
                    init_open_hstate=open_init_hstate.squeeze(1),
                    advantages=advantages,
                    targets=targets,
                    clip_eps=config["CLIP_EPS"],
                    vf_coef=config["VF_COEF"],
                    ent_coef=config["ENT_COEF"],
                    update_target_counter=update_target_counter,
                    ema_param=config["EMA_PARAMETER"],
                )
                return (
                    new_train_state,
                    pred_state,
                    target_state,
                    update_target_counter,
                ), update_info

            (
                rng,
                train_state,
                pred_state,
                target_state,
                update_target_counter,
                init_hstate,
                close_init_hstate,
                open_init_hstate,
                transitions,
                advantages,
                targets,
            ) = update_state

            # MINIBATCHES PREPARATION
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, config["NUM_ENVS_PER_DEVICE"])
            # [seq_len, batch_size, ...]
            batch = (
                init_hstate,
                close_init_hstate,
                open_init_hstate,
                transitions,
                advantages,
                targets,
            )
            # [batch_size, seq_len, ...], as our model assumes
            batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

            shuffled_batch = jtu.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
            # [num_minibatches, minibatch_size, ...]
            minibatches = jtu.tree_map(
                lambda x: jnp.reshape(x, (config["NUM_MINIBATCHES"], -1) + x.shape[1:]),
                shuffled_batch,
            )
            (
                train_state,
                pred_state,
                target_state,
                update_target_counter,
            ), update_info = jax.lax.scan(
                _update_minbatch,
                (train_state, pred_state, target_state, update_target_counter),
                minibatches,
            )

            update_state = (
                rng,
                train_state,
                pred_state,
                target_state,
                update_target_counter,
                init_hstate,
                close_init_hstate,
                open_init_hstate,
                transitions,
                advantages,
                targets,
            )
            return update_state, update_info

        # [seq_len, batch_size, num_layers, hidden_dim]
        init_hstate = initial_hstate[None, :]
        close_init_hstate = close_initial_hstate[None, :]
        open_init_hstate = open_initial_hstate[None, :]
        update_state = (
            rng,
            train_state,
            pred_state,
            target_state,
            update_target_counter,
            init_hstate,
            close_init_hstate,
            open_init_hstate,
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

        rng, train_state, pred_state, target_state, update_target_counter = update_state[:5]
        traj_batch = update_state[-3]
        metric = traj_batch.info
        int_reward = traj_batch.int_reward
        runner_state = (
            rng,
            train_state,
            pred_state,
            target_state,
            env_state,
            prev_obs,
            prev_action,
            prev_reward,
            prev_bt,
            hstate,
            close_hstate,
            open_hstate,
            byol_reward_norm_params,
            update_target_counter,
        )
        return runner_state, (metric, loss_info, int_reward, norm_int_reward)

    runner_state = (
        rng,
        train_state,
        pred_state,
        target_state,
        env_state,
        obsv,
        prev_action,
        prev_reward,
        prev_bt,
        init_hstate,
        close_init_hstate,
        open_init_hstate,
        byol_reward_norm_params,
        update_target_counter,
    )
    runner_state, (metric, loss, int_reward, _) = jax.lax.scan(
        _update_step, runner_state, None, config["NUM_UPDATES"]
    )
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
        (
            init_hstate,
            close_init_hstate,
            open_init_hstate,
            train_state,
            pred_state,
            target_state,
            rng,
        ) = jax.jit(jax.vmap(make_train, out_axes=(0, 0, 0, 0, 0, 0, 1)))(rng)
        init_hstate = replicate(init_hstate, jax.local_devices())

        open_init_hstate = replicate(open_init_hstate, jax.local_devices())
        close_init_hstate = replicate(close_init_hstate, jax.local_devices())
        train_state = replicate(train_state, jax.local_devices())
        pred_state = replicate(pred_state, jax.local_devices())
        target_state = replicate(target_state, jax.local_devices())

        train_fn = jax.vmap(train)
        train_fn = jax.pmap(train_fn, axis_name="devices")
        print(f"Training in {config['ENV_NAME']}")
        t = time.time()
        output = jax.block_until_ready(
            train_fn(
                rng,
                init_hstate,
                close_init_hstate,
                open_init_hstate,
                train_state,
                pred_state,
                target_state,
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
            open_init_hstate,
            close_init_hstate,
            pred_state,
            target_state,
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
    start_time = time.time()

    # Ensure the save directory exists

    # Load random agent data and calculate the mean
    random_agent_returns = np.load(random_agent_path)
    print(f"Random Last Episode Return: {random_agent_returns}")

    random_agent_mean = random_agent_returns.mean()
    print(f"Random Agent Mean: {random_agent_mean}")

    # Load baseline data and get the last element (episode return)
    baseline_returns = np.load(baseline_path)
    baseline_last_episode_return = baseline_returns[-1]
    print(f"Baseline Last Episode Return: {baseline_last_episode_return}")
    print(f"Baseline Last Episode Return SHape: {baseline_last_episode_return.shape}")
    print(f"Baseline Mean: {baseline_last_episode_return.mean()}")

    # Load curious agent data and get the last element (episode return)
    print(f"Curious Agent Last Episode Return: {curious_agent_last_episode_return}")
    print(f"Curious Agent Last Episode Return Shape: {curious_agent_last_episode_return.shape}")

    # Normalize the curious agent returns between 0 and 1
    normalized_curious_agent_returns = (curious_agent_last_episode_return - random_agent_mean) / (
        baseline_last_episode_return.mean() - random_agent_mean
    )
    print(f"Normalized Curious Agent Returns: {normalized_curious_agent_returns}")

    # Construct the save path using the curious algorithm type
    # os.makedirs(save_path, exist_ok=True)

    # Save the normalized curious agent returns
    # normalized_curious_agent_file = os.path.join(save_path, "normalized_curious_agent_returns.npy")
    # np.save(normalized_curious_agent_file, normalized_curious_agent_returns)

    # Print the size of the saved file in MB
    # print(
    #     f"Size of normalized_curious_agent_returns.npy: {os.path.getsize(normalized_curious_agent_file) / (1024 * 1024):.2f} MB"
    # )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to run the code: {elapsed_time:.2f} seconds")

    return normalized_curious_agent_returns


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

# Scatter plot for final episode returns vs. lambda values
for env_name in environments:
    lambda_values = []
    final_returns = []

    # Collect data for plotting
    for lambda_value, env_data in y_values.items():
        epi_ret = env_data[env_name][0]  # index 0 for episode returns
        epi_ret = np.array(env_data[env_name][0])
        if epi_ret.size > 0:  # Ensure there is at least one return value
            final_returns.append(epi_ret)
            lambda_values.append(lambda_value)

    # Sort lambda values for consistent plotting
    sorted_indices = sorted(range(len(lambda_values)), key=lambda k: lambda_values[k])
    sorted_lambda_values = [lambda_values[i] for i in sorted_indices]
    sorted_final_returns = [final_returns[i] for i in sorted_indices]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(range(1, len(sorted_lambda_values) + 1), sorted_final_returns, color="blue")
    plt.title(f"Final Episode Returns vs. Lambda for {env_name}")
    plt.xlabel("Lambda Value (Indexed)")
    plt.ylabel("Final Episode Return")
    plt.xticks(
        range(1, len(sorted_lambda_values) + 1),
        [f"{lv:.5f}" for lv in sorted_lambda_values],
        rotation=45,
    )
    plt.grid(True)

    # Save and close plot
    plt.savefig(
        f"MetaLearnCuriosity/hyperparameter_sweep/MiniGrid/{env_name}_final_episode_returns_vs_lambda_byol.png"
    )
    plt.close()  # Close the plot to free up memory
    gc.collect()

for lambda_value, env_data in y_values.items():
    for env_name, epi_ret in env_data.items():
        # Assuming you have precomputed normalization_factors for each env_name

        y_values[lambda_value][env_name] = normalize_curious_agent_returns(
            f"/home/batsy/MetaLearnCuriosity/MetaLearnCuriosity/experiments/baseline/{env_name}/metric_seeds_episode_return.npy",
            f"/home/batsy/MetaLearnCuriosity/MetaLearnCuriosity/experiments/random_agents/{env_name}_epi_rets.npy",
            epi_ret.T,
        )

for lambda_value, env_data in y_values.items():
    # Create a list to store all normalized epi_ret for this lambda_value
    normalized_returns = []

    # Collect normalized returns for each environment
    for env_name, normalized_epi_ret in env_data.items():
        normalized_returns.append(normalized_epi_ret)

    # Convert the list to a NumPy array (this is more memory-efficient for large data)
    y_values[lambda_value] = np.array(normalized_returns)
    # np.save(f"y_values_{lambda_value}.npy", np.array(normalized_returns))

# Collect lambda values and normalized returns
lambda_values = sorted(y_values.keys())
means = []
ci_lows = []
ci_highs = []

# Calculate means and CIs
for lambda_value in lambda_values:
    normalized_returns = np.array(y_values[lambda_value]).flatten()
    mean_value = np.mean(normalized_returns)
    means.append(mean_value)

    ci = bootstrap(
        (normalized_returns,),
        np.mean,
        confidence_level=0.95,
        method="percentile",
    )
    ci_lows.append(ci.confidence_interval.low)
    ci_highs.append(ci.confidence_interval.high)

# Create evenly spaced x positions
x_positions = np.arange(len(lambda_values))

plt.figure(figsize=(15, 8))  # Maintain larger figure size for clarity

# Create the bar plot
plt.bar(
    x_positions,
    means,
    yerr=[np.array(means) - np.array(ci_lows), np.array(ci_highs) - np.array(means)],
    capsize=5,
    color="skyblue",
    edgecolor="black",
    width=0.6,
)

# Add grid
plt.grid(True)

# Set x-axis ticks and labels
plt.xticks(x_positions, [f"{lv:.4f}" for lv in lambda_values], rotation=45, ha="right")

# Adjust y-axis to show both positive and negative values
y_min = min(min(ci_lows), 0)
y_max = max(ci_highs)
plt.ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))

# Add a horizontal line at y=0
plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

# Label the axes
plt.xlabel("Lambda Value", fontsize=12)
plt.ylabel("Mean Normalized Episode Return", fontsize=12)
plt.title("Mean Normalized Episode Return vs Lambda Value", fontsize=14)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig(
    "/home/batsy/MetaLearnCuriosity/MetaLearnCuriosity/hyperparameter_sweep/MiniGrid/MiniGrid_normalized_returns_histogram.png",
    dpi=300,
)
plt.close()
