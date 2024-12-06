import os
import shutil
import time
from typing import Any, NamedTuple, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.jax_utils import replicate, unreplicate
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from MetaLearnCuriosity.agents.nn import (
    RCRNN,
    BraxBYOLPredictor,
    BYOLTarget,
    CloseScannedRNN,
    OpenScannedRNN,
    RewardCombiner,
)
from MetaLearnCuriosity.checkpoints import Save
from MetaLearnCuriosity.logger import WBLogger
from MetaLearnCuriosity.utils import BYOLRewardNorm
from MetaLearnCuriosity.utils import RCBYOLTransition as Transition
from MetaLearnCuriosity.utils import (
    byol_normalize_prior_int_rewards,
    compress_output_for_reasoning,
    process_output_general,
    update_target_state_with_ema,
)
from MetaLearnCuriosity.wrappers import (
    BraxGymnaxWrapper,
    ClipAction,
    DelayedReward,
    LogWrapper,
    NormalizeVecObservation,
    NormalizeVecReward,
    VecEnv,
)

environments = [
    # "ant",
    # "halfcheetah",
    # "hopper",
    # "humanoid",
    # "humanoidstandup",
    "inverted_pendulum",
    # "inverted_double_pendulum",
    # "pusher",
    # "reacher",
    # "walker2d",
]

config = {
    "RUN_NAME": "rc_delayed_brax_byol",
    "SEED": 42,
    "NUM_SEEDS": 30,
    "LR": 3e-4,
    "NUM_ENVS": 2048,
    "NUM_STEPS": 10,  # unroll length
    "TOTAL_TIMESTEPS": 5e7,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 32,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.0,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ANNEAL_LR": False,
    "NORMALIZE_ENV": True,
    "DELAY_REWARDS": True,
    "ANNEAL_PRED_LR": False,
    "DEBUG": False,
    "PRED_LR": 0.001,
    "REW_NORM_PARAMETER": 0.99,
    "EMA_PARAMETER": 0.99,
}

step_intervals = [3, 10, 20, 30]


class PPOActorCritic(nn.Module):
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


def compile_brax_byol_fns(config):  # noqa: C901
    def make_config_env(config, env_name):
        config["ENV_NAME"] = env_name
        env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
        num_devices = jax.local_device_count()
        assert config["NUM_ENVS"] % num_devices == 0
        config["NUM_ENVS_PER_DEVICE"] = config["NUM_ENVS"] // num_devices
        config["TOTAL_TIMESTEPS_PER_DEVICE"] = config["TOTAL_TIMESTEPS"] // num_devices
        # config["EVAL_EPISODES_PER_DEVICE"] = config["EVAL_EPISODES"] // num_devices
        config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS_PER_DEVICE"]
            // config["NUM_STEPS"]
            // config["NUM_ENVS_PER_DEVICE"]
        )
        config["MINIBATCH_SIZE"] = (
            config["NUM_ENVS_PER_DEVICE"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
        )
        config["TRAINING_HORIZON"] = (
            config["TOTAL_TIMESTEPS_PER_DEVICE"] // config["NUM_ENVS_PER_DEVICE"]
        )
        env = LogWrapper(env)
        env = ClipAction(env)
        if config["DELAY_REWARDS"]:
            env = DelayedReward(env, config["STEP_INTERVAL"])
        env = VecEnv(env)
        if config["NORMALIZE_ENV"]:
            env = NormalizeVecObservation(env)
            env = NormalizeVecReward(env, config["GAMMA"])

        return config, env, env_params

    def ppo_make_train(rng):
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
            )
            return config["LR"] * frac

        def pred_linear_schedule(count):
            frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
            )
            return config["PRED_LR"] * frac

        # INIT NETWORK
        network = PPOActorCritic(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        target = BYOLTarget(256)
        pred = BraxBYOLPredictor(256)

        # KEYS
        rng, _rng = jax.random.split(rng)
        rng, _tar_rng = jax.random.split(rng)
        # rng, _en_rng = jax.random.split(rng)
        rng, _pred_rng = jax.random.split(rng)

        # INIT INPUT
        init_x = jnp.zeros(
            (1, config["NUM_ENVS_PER_DEVICE"], *env.observation_space(env_params).shape)
        )
        init_action = jnp.zeros(
            (config["NUM_ENVS_PER_DEVICE"], *env.action_space(env_params).shape), dtype=jnp.float32
        )
        close_init_hstate = CloseScannedRNN.initialize_carry(config["NUM_ENVS_PER_DEVICE"], 256)
        open_init_hstate = OpenScannedRNN.initialize_carry(config["NUM_ENVS_PER_DEVICE"], 256)
        init_bt = jnp.zeros((1, config["NUM_ENVS_PER_DEVICE"], 256))
        ext_reward_history = jnp.zeros((config["NUM_ENVS_PER_DEVICE"], config["HIST_LEN"]))
        int_reward_history = jnp.zeros((config["NUM_ENVS_PER_DEVICE"], config["HIST_LEN"]))

        init_pred_input = (init_bt, init_x, init_action[np.newaxis, :], init_action[np.newaxis, :])

        network_params = network.init(_rng, init_x)
        pred_params = pred.init(_pred_rng, close_init_hstate, open_init_hstate, init_pred_input)
        target_params = target.init(_tar_rng, init_x)

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

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
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

        rng = jax.random.split(rng, jax.local_device_count())

        return (
            rng,
            train_state,
            pred_state,
            target_state,
            init_bt,
            close_init_hstate,
            open_init_hstate,
            init_action,
            ext_reward_history,
            int_reward_history,
        )

    def train(
        rng,
        rc_params,
        train_state,
        pred_state,
        target_state,
        init_bt,
        close_init_hstate,
        open_init_hstate,
        init_action,
        ext_reward_hist,
        int_reward_hist,
    ):
        rc_network = RewardCombiner()

        # INIT STUFF FOR OPTIMIZATION AND NORMALIZATION
        update_target_counter = 0
        byol_reward_norm_params = BYOLRewardNorm(0, 0, 1, 0)
        ext_reward_norm_params = BYOLRewardNorm(0, 0, 1, 0)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS_PER_DEVICE"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, step_index):
                (
                    train_state,
                    pred_state,
                    target_state,
                    bt,
                    close_hstate,
                    open_hstate,
                    last_act,
                    env_state,
                    last_obs,
                    byol_reward_norm_params,
                    ext_reward_norm_params,
                    update_target_counter,
                    ext_reward_hist,
                    int_reward_hist,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = train_state.apply_fn(train_state.params, last_obs[np.newaxis, :])
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS_PER_DEVICE"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action.squeeze(0), env_params
                )

                # INT REWARD
                tar_obs = target_state.apply_fn(target_state.params, obsv[np.newaxis, :])
                pred_input = (bt, last_obs[np.newaxis, :], last_act[np.newaxis, :], action)
                pred_obs, new_bt, new_close_hstate, new_open_hstate = pred_state.apply_fn(
                    pred_state.params, close_hstate, open_hstate, pred_input
                )
                pred_norm = (pred_obs.squeeze(0)) / (
                    jnp.linalg.norm(pred_obs.squeeze(0), axis=-1, keepdims=True)
                )
                tar_norm = jax.lax.stop_gradient(
                    (tar_obs.squeeze(0))
                    / (jnp.linalg.norm(tar_obs.squeeze(0), axis=-1, keepdims=True))
                )
                int_reward = jnp.square(jnp.linalg.norm((pred_norm - tar_norm), axis=-1)) * (
                    1 - done
                )
                value, action, log_prob = (value.squeeze(0), action.squeeze(0), log_prob.squeeze(0))
                norm_time_step = info["timestep"] / config["TRAINING_HORIZON"]

                ext_reward_hist = jnp.roll(ext_reward_hist, shift=-1, axis=1)
                int_reward_hist = jnp.roll(int_reward_hist, shift=-1, axis=1)
                ext_reward_hist = ext_reward_hist.at[:, -1].set(reward)
                int_reward_hist = int_reward_hist.at[:, -1].set(int_reward)

                transition = Transition(
                    done,
                    last_act,
                    action,
                    value,
                    reward,
                    reward,
                    int_reward,
                    log_prob,
                    last_obs,
                    obsv,
                    bt,
                    norm_time_step,
                    ext_reward_hist,
                    int_reward_hist,
                    info,
                )

                runner_state = (
                    train_state,
                    pred_state,
                    target_state,
                    new_bt,
                    new_close_hstate,
                    new_open_hstate,
                    action,
                    env_state,
                    obsv,
                    byol_reward_norm_params,
                    ext_reward_norm_params,
                    update_target_counter,
                    ext_reward_hist,
                    int_reward_hist,
                    rng,
                )
                return runner_state, transition

            close_initial_hstate, open_initial_hstate = runner_state[4:6]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, jnp.arange(config["NUM_STEPS"])
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                pred_state,
                target_state,
                bt,
                close_hstate,
                open_hstate,
                last_act,
                env_state,
                last_obs,
                byol_reward_norm_params,
                ext_reward_norm_params,
                update_target_counter,
                ext_reward_hist,
                int_reward_hist,
                rng,
            ) = runner_state

            # update_target_counter+=1
            _, last_val = train_state.apply_fn(train_state.params, last_obs[np.newaxis, :])

            def _calculate_gae(
                traj_batch,
                last_val,
                byol_reward_norm_params,
                ext_reward_norm_params,
            ):
                (
                    norm_int_reward,
                    byol_reward_norm_params,
                    int_reward_hist,
                ) = byol_normalize_prior_int_rewards(
                    traj_batch.int_reward,
                    byol_reward_norm_params,
                    config["REW_NORM_PARAMETER"],
                    traj_batch.int_reward_hist,
                )
                (
                    norm_ext_reward,
                    ext_reward_norm_params,
                    ext_reward_hist,
                ) = byol_normalize_prior_int_rewards(
                    traj_batch.norm_reward,
                    ext_reward_norm_params,
                    config["REW_NORM_PARAMETER"],
                    traj_batch.ext_reward_hist,
                    prior=False,
                )
                norm_traj_batch = Transition(
                    traj_batch.done,
                    traj_batch.prev_action,
                    traj_batch.action,
                    traj_batch.value,
                    traj_batch.reward,
                    traj_batch.reward,
                    norm_int_reward,
                    traj_batch.log_prob,
                    traj_batch.obs,
                    traj_batch.next_obs,
                    traj_batch.bt,
                    traj_batch.norm_time_step,
                    ext_reward_hist,
                    int_reward_hist,
                    traj_batch.info,
                )

                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward, int_reward, _, _, ext_reward_hist, int_reward_hist = (
                        transition.done,
                        transition.value,
                        transition.reward,
                        transition.int_reward,
                        transition.norm_time_step,
                        transition.norm_reward,
                        transition.ext_reward_hist,
                        transition.int_reward_hist,
                    )
                    rc_input = jnp.stack(
                        (ext_reward_hist, int_reward_hist),
                        axis=-1,
                    )
                    int_lambda = rc_network.apply(rc_params, rc_input)
                    delta = (
                        (reward + (int_reward * int_lambda))
                        + config["GAMMA"] * next_value * (1 - done)
                        - value
                    )
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), (gae, int_lambda)

                (_, _), (advantages, int_lambdas) = jax.lax.scan(
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
                    byol_reward_norm_params,
                    ext_reward_norm_params,
                    int_lambdas,
                )

            (
                advantages,
                targets,
                norm_int_reward,
                byol_reward_norm_params,
                ext_reward_norm_params,
                int_lambdas,
            ) = _calculate_gae(
                traj_batch,
                last_val.squeeze(0),
                byol_reward_norm_params,
                ext_reward_norm_params,
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    (
                        traj_batch,
                        advantages,
                        targets,
                        init_close_hstate,
                        init_open_hstate,
                    ) = batch_info
                    train_state, pred_state, target_state, update_target_counter = train_states

                    def pred_loss(
                        pred_params, target_params, traj_batch, init_close_hstate, init_open_hstate
                    ):
                        tar_obs = target_state.apply_fn(target_params, traj_batch.next_obs)
                        pred_input = (
                            traj_batch.bt,
                            traj_batch.obs,
                            traj_batch.prev_action,
                            traj_batch.action,
                        )
                        pred_obs, _, _, _ = pred_state.apply_fn(
                            pred_params, init_close_hstate[0], init_open_hstate[0], pred_input
                        )
                        pred_norm = (pred_obs) / (jnp.linalg.norm(pred_obs, axis=-1, keepdims=True))
                        tar_norm = jax.lax.stop_gradient(
                            (tar_obs) / (jnp.linalg.norm(tar_obs, axis=-1, keepdims=True))
                        )
                        loss = jnp.square(jnp.linalg.norm((pred_norm - tar_norm), axis=-1)) * (
                            1 - traj_batch.done
                        )
                        return loss.mean()

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
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    (loss, (vloss, aloss, entropy)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params, traj_batch, advantages, targets)
                    pred_losses, pred_grads = jax.value_and_grad(pred_loss)(
                        pred_state.params,
                        target_state.params,
                        traj_batch,
                        init_close_hstate,
                        init_open_hstate,
                    )
                    (loss, vloss, aloss, entropy, pred_losses, grads, pred_grads) = jax.lax.pmean(
                        (loss, vloss, aloss, entropy, pred_losses, grads, pred_grads),
                        axis_name="devices",
                    )

                    def update_target(
                        target_state, pred_state, update_target_counter=update_target_counter
                    ):
                        def true_fun(_):
                            # Perform the EMA update
                            return update_target_state_with_ema(
                                predictor_state=pred_state,
                                target_state=target_state,
                                ema_param=config["EMA_PARAMETER"],
                            )

                        def false_fun(_):
                            # Return the old target_params unchanged
                            return target_state

                        # Conditionally update every 10 steps
                        return jax.lax.cond(
                            update_target_counter
                            % (10 * config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
                            == 0,
                            true_fun,
                            false_fun,
                            None,  # The argument passed to true_fun and false_fun, `_` in this case is unused
                        )

                    update_target_counter += 1
                    train_state = train_state.apply_gradients(grads=grads)
                    pred_state = pred_state.apply_gradients(grads=pred_grads)
                    target_state = update_target(target_state, pred_state, update_target_counter)

                    return (train_state, pred_state, target_state, update_target_counter), (
                        loss,
                        (vloss, aloss, entropy),
                        pred_losses,
                    )

                (
                    train_state,
                    pred_state,
                    target_state,
                    update_target_counter,
                    init_close_hstate,
                    init_open_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS_PER_DEVICE"])
                batch = (traj_batch, advantages, targets, init_close_hstate, init_open_hstate)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )
                (
                    train_state,
                    pred_state,
                    target_state,
                    update_target_counter,
                ), total_loss = jax.lax.scan(
                    _update_minbatch,
                    (train_state, pred_state, target_state, update_target_counter),
                    minibatches,
                )
                update_state = (
                    train_state,
                    pred_state,
                    target_state,
                    update_target_counter,
                    init_close_hstate,
                    init_open_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            traj_batch = Transition(
                traj_batch.done,
                traj_batch.prev_action,
                traj_batch.action,
                traj_batch.value,
                traj_batch.reward,
                traj_batch.reward,
                traj_batch.int_reward,
                traj_batch.log_prob,
                traj_batch.obs,
                traj_batch.next_obs,
                traj_batch.bt.squeeze(1),
                traj_batch.norm_time_step,
                traj_batch.ext_reward_hist,
                traj_batch.int_reward_hist,
                traj_batch.info,
            )

            update_state = (
                train_state,
                pred_state,
                target_state,
                update_target_counter,
                open_initial_hstate[np.newaxis, :],
                close_initial_hstate[np.newaxis, :],
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state, pred_state, target_state, update_target_counter = update_state[:4]
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
                target_state,
                bt,
                close_hstate,
                open_hstate,
                last_act,
                env_state,
                last_obs,
                byol_reward_norm_params,
                ext_reward_norm_params,
                update_target_counter,
                ext_reward_hist,
                int_reward_hist,
                rng,
            )
            return runner_state, (
                metric,
                loss_info,
                traj_batch.int_reward,
                norm_int_reward,
                int_lambdas,
            )

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            pred_state,
            target_state,
            init_bt,
            close_init_hstate,
            open_init_hstate,
            init_action,
            env_state,
            obsv,
            byol_reward_norm_params,
            ext_reward_norm_params,
            update_target_counter,
            ext_reward_hist,
            int_reward_hist,
            _rng,
        )
        runner_state, extra_info = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        metric, _, _, _, int_lambdas = extra_info
        rewards = metric["sum_of_rewards"].mean(axis=-1)
        rewards = rewards.reshape(-1)
        rewards = rewards[-1]

        lifetime_episode_lengths = metric["lifetime_episode_lengths"].mean(axis=-1)
        lifetime_episode_lengths = lifetime_episode_lengths.reshape(-1)
        lifetime_episode_lengths = lifetime_episode_lengths[-1]

        int_lambdas = int_lambdas.mean()
        episode_returns = metric["returned_episode_returns"].mean(axis=-1)
        episode_returns = episode_returns.reshape(-1)
        episode_returns = episode_returns[-1]
        return {"rewards": rewards, "int_lambdas": int_lambdas, "episode_returns": episode_returns}

    train_fns = {}
    make_seeds = {}
    env_name = "inverted_pendulum"
    for step_interval in step_intervals:
        config["STEP_INTERVAL"] = step_interval
        config, env, env_params = make_config_env(config, env_name)
        make_train = jax.vmap(ppo_make_train, out_axes=(1, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        train_fn = jax.vmap(train, in_axes=(0, None, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        train_fn = jax.vmap(
            train_fn,
            in_axes=(
                None,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )
        train_fn = jax.pmap(train_fn, axis_name="devices")
        train_fns[step_interval] = train_fn
        make_seeds[step_interval] = make_train

    return train_fns, make_seeds
