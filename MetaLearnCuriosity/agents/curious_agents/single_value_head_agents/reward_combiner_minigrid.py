# Taken from:
# https://github.com/corl-team/xland-minigrid/blob/main/training/train_single_task.py

import os
import shutil
import time

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState

import wandb
from MetaLearnCuriosity.agents.nn import (
    BYOLTarget,
    MiniGridActorCriticRNN,
    MiniGridBYOLPredictor,
)
from MetaLearnCuriosity.checkpoints import Restore, Save
from MetaLearnCuriosity.logger import WBLogger
from MetaLearnCuriosity.utils import BYOLRewardNorm
from MetaLearnCuriosity.utils import RCBYOLMiniGridTransition as Transition
from MetaLearnCuriosity.utils import (
    byol_minigrid_ppo_update_networks,
    process_output_general,
    rc_byol_calculate_gae,
    rnn_rollout,
)
from MetaLearnCuriosity.wrappers import (
    FlattenObservationWrapper,
    LogWrapper,
    MiniGridGymnax,
    VecEnv,
)

jax.config.update("jax_threefry_partitionable", True)

environments = [
    "MiniGrid-BlockedUnlockPickUp",
    "MiniGrid-Empty-16x16",
    "MiniGrid-EmptyRandom-16x16",
    "MiniGrid-FourRooms",
    "MiniGrid-MemoryS128",
    "MiniGrid-Unlock",
]

config = {
    "NUM_SEEDS": 10,
    "PROJECT": "MetaLearnCuriosity",
    "RUN_NAME": "rc_default_minigrid_byol",
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
    # "INT_LAMBDA": 0.0003,
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
    config["TRAINING_HORIZON"] = (
        config["TOTAL_TIMESTEPS_PER_DEVICE"] // config["NUM_ENVS_PER_DEVICE"]
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
    init_pred_input = (init_bt, init_x, init_action, init_action)
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
    rng,
    rc_params,
    init_hstate,
    close_init_hstate,
    open_init_hstate,
    train_state,
    pred_state,
    target_state,
):
    rng, _rng = jax.random.split(rng)

    reset_rng = jax.random.split(_rng, config["NUM_ENVS_PER_DEVICE"])
    # INIT STUFF FOR OPTIMIZATION AND NORMALIZATION
    update_target_counter = 0
    byol_reward_norm_params = BYOLRewardNorm(0, 0, 1, 0)
    ext_reward_norm_params = BYOLRewardNorm(0, 0, 1, 0)

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
                ext_reward_norm_params,
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
            norm_time_step = info["timestep"] / config["TRAINING_HORIZON"]

            # INT REWARD
            tar_obs = target_state.apply_fn(target_state.params, obsv[:, None])
            pred_input = (prev_bt, prev_obs[:, None], prev_action[:, None], action[:, None])
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
                norm_reward=reward,
                int_reward=int_reward,
                log_prob=log_prob,
                obs=prev_obs,
                next_obs=obsv,
                prev_action=prev_action,
                prev_reward=prev_reward,
                prev_bt=prev_bt,
                norm_time_step=norm_time_step,
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
                ext_reward_norm_params,
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
            ext_reward_norm_params,
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

        (
            advantages,
            targets,
            norm_int_reward,
            norm_ext_reward,
            byol_reward_norm_params,
            ext_reward_norm_params,
            int_lambda,
        ) = rc_byol_calculate_gae(
            transitions,
            rc_params,
            last_val.squeeze(1),
            config["GAMMA"],
            config["GAE_LAMBDA"],
            config["REW_NORM_PARAMETER"],
            byol_reward_norm_params,
            ext_reward_norm_params,
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
        reward = traj_batch.reward
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
            ext_reward_norm_params,
            update_target_counter,
        )
        return runner_state, (
            metric,
            loss_info,
            int_reward,
            norm_int_reward,
            reward,
            norm_ext_reward,
            int_lambda,
        )

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
        ext_reward_norm_params,
        update_target_counter,
    )
    runner_state, (
        metric,
        _,
        int_reward,
        norm_int_reward,
        reward,
        norm_ext_reward,
        int_lambda,
    ) = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])

    return {
        # "train_states": runner_state[1:4],
        "metrics": metric,
        # "loss_info": loss,
        # "rl_total_loss": loss["total_loss"],
        # "rl_value_loss": loss["value_loss"],
        # "rl_actor_loss": loss["actor_loss"],
        # "rl_entrophy_loss": loss["entropy"],
        "int_reward": int_reward,
        "norm_int_reward": norm_int_reward,
        # "pred_loss": loss["pred_loss"],
        "int_lambdas": int_lambda,
        "reward": reward,
        "norm_ext_reward": norm_ext_reward,
    }


for env_name in environments:
    rc_params = Restore(
        "/home/batsy/MetaLearnCuriosity/rc_meta_default_SpaceInvaders-MinAtar_flax-checkpoints_v0"
    )
    observations_shape, config, env, env_params = make_env_config(config, env_name)

    # experiments
    rng = jax.random.PRNGKey(config["SEED"])

    if config["NUM_SEEDS"] > 1:
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
        rc_params = replicate(rc_params, jax.local_devices())

        train_fn = jax.vmap(train, in_axes=(0, None, 0, 0, 0, 0, 0, 0))
        train_fn = jax.pmap(train_fn, axis_name="devices")
        print(f"Training in {config['ENV_NAME']}")
        t = time.time()
        output = jax.block_until_ready(
            train_fn(
                rng,
                rc_params,
                init_hstate,
                close_init_hstate,
                open_init_hstate,
                train_state,
                pred_state,
                target_state,
            )
        )
        elapsed_time = time.time() - t

    else:
        (
            init_hstate,
            close_init_hstate,
            open_init_hstate,
            train_state,
            pred_state,
            target_state,
            rng,
        ) = make_train(rng)
        init_hstate = replicate(init_hstate, jax.local_devices())
        open_init_hstate = replicate(open_init_hstate, jax.local_devices())
        close_init_hstate = replicate(close_init_hstate, jax.local_devices())
        train_state = replicate(train_state, jax.local_devices())
        pred_state = replicate(pred_state, jax.local_devices())
        target_state = replicate(target_state, jax.local_devices())
        train_fn = jax.pmap(train, axis_name="devices")
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

    logger = WBLogger(
        config=config,
        group="delayed_brax_curious",
        tags=["reward_combiner", config["ENV_NAME"], "delayed_brax"],
        name=f'{config["RUN_NAME"]}_{config["ENV_NAME"]}',
    )
    output = process_output_general(output)

    logger.log_reward(output, config["NUM_SEEDS"])
    logger.log_norm_ext_rewards(output, config["NUM_SEEDS"])
    logger.log_int_lambdas(output, config["NUM_SEEDS"])
    logger.log_episode_return(output, config["NUM_SEEDS"])
    logger.log_int_rewards(output, config["NUM_SEEDS"])
    logger.log_norm_int_rewards(output, config["NUM_SEEDS"])
    output["config"] = config
    checkpoint_directory = f'MLC_logs/flax_ckpt/{config["ENV_NAME"]}/{config["RUN_NAME"]}'

    # Get the absolute path of the directory
    episode_returns = {}
    episode_returns["returned_episode_returns"] = output["metrics"]["returned_episode_returns"]
    path = os.path.abspath(checkpoint_directory)
    Save(path, episode_returns)
    logger.save_artifact(path)
    shutil.rmtree(path)
    print(f"Deleted local checkpoint directory: {path}")
    print(f"Done in {elapsed_time / 60:.2f}min")
