# Taken from:
# https://github.com/corl-team/xland-minigrid/blob/main/training/train_single_task.py

import os
import shutil
import time
from typing import Sequence

import distrax
import flax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import optax
from flax.jax_utils import replicate, unreplicate
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from tqdm import tqdm

import wandb
from MetaLearnCuriosity.agents.nn import (
    PredictorNetwork,
    TargetNetwork,
    TemporalRewardCombiner,
)
from MetaLearnCuriosity.checkpoints import Save
from MetaLearnCuriosity.logger import WBLogger
from MetaLearnCuriosity.pmapped_open_es import OpenES
from MetaLearnCuriosity.utils import ObsNormParams
from MetaLearnCuriosity.utils import RCRNDTransition as RNDTransition
from MetaLearnCuriosity.utils import (
    RNDNormIntReturnParams,
    make_obs_gymnax_discrete,
    process_output_general,
    rnd_normalise_ext_rewards,
    rnd_normalise_int_rewards,
    update_obs_norm_params,
)
from MetaLearnCuriosity.wrappers import FlattenObservationWrapper, LogWrapper, VecEnv

environments = [
    # "Asterix-MinAtar",
    "Breakout-MinAtar",
    # "Freeway-MinAtar",
    # "SpaceInvaders-MinAtar",
]


class PPOActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            actor_mean
        )
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


config = {
    "RUN_NAME": "rc_breakout_rnd_default",
    "SEED": 42,
    "NUM_SEEDS": 3,
    "LR": 5e-3,
    "NUM_ENVS": 64,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 1e7,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 8,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu",
    "ANNEAL_LR": True,
    "ANNEAL_PRED_LR": False,
    "DEBUG": False,
    "PRED_LR": 1e-3,
    "REW_NORM_PARAMETER": 0.99,
    "EMA_PARAMETER": 0.99,
    "POP_SIZE": 32,
    "ES_SEED": 7,
    "RC_SEED": 23,
    "NUM_GENERATIONS": 128,
    # "INT_LAMBDA": 0.001,
}


def make_config_env(config, env_name):
    config["ENV_NAME"] = env_name
    num_devices = jax.local_device_count()
    assert config["NUM_ENVS"] % num_devices == 0
    config["NUM_ENVS_PER_DEVICE"] = config["NUM_ENVS"] // 1
    config["TOTAL_TIMESTEPS_PER_DEVICE"] = config["TOTAL_TIMESTEPS"] // 1
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
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    env = VecEnv(env)

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

    # INIT NETWORKS
    network = PPOActorCritic(env.action_space(env_params).n, activation=config["ACTIVATION"])
    target = TargetNetwork(64)
    predictor = PredictorNetwork(64)

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
        optax.adam(pred_linear_schedule, eps=1e-5),
    )

    predictor_state = TrainState.create(apply_fn=predictor.apply, params=pred_params, tx=pred_tx)

    rng = jax.random.split(rng, jax.local_device_count())

    return rng, train_state, predictor_state, target_params, _init_obs_rng


def train(rng, rc_params, train_state, pred_state, target_params, init_obs_rng):
    # REWARD COMBINER
    rc_network = TemporalRewardCombiner()

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
    rnd_ext_return_norm_params = RNDNormIntReturnParams(
        1e-4, 0.0, 1.0, jnp.zeros((config["NUM_STEPS"],))
    )

    obs_norm_params = update_obs_norm_params(init_obs_norm_params, init_obs)
    target = TargetNetwork(64)

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
                rnd_ext_return_norm_params,
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

            # Norm time step
            norm_time_step = info["timestep"] / config["TRAINING_HORIZON"]

            transition = RNDTransition(
                done,
                action,
                value,
                reward,
                reward,
                int_reward,
                log_prob,
                last_obs,
                norm_time_step,
                info,
            )
            runner_state = (
                train_state,
                pred_state,
                target_params,
                env_state,
                obsv,
                rnd_int_return_norm_params,
                rnd_ext_return_norm_params,
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
            rnd_ext_return_norm_params,
            obs_norm_params,
            rng,
        ) = runner_state
        _, last_val = train_state.apply_fn(train_state.params, last_obs)

        def _calculate_gae(
            traj_batch, last_val, rnd_int_return_norm_params, rnd_ext_return_norm_params
        ):

            norm_int_reward, rnd_int_return_norm_params = rnd_normalise_int_rewards(
                traj_batch, rnd_int_return_norm_params, config["INT_GAMMA"]
            )
            norm_ext_reward, rnd_ext_return_norm_params = rnd_normalise_ext_rewards(
                traj_batch, rnd_ext_return_norm_params, config["GAMMA"]
            )

            norm_traj_batch = RNDTransition(
                traj_batch.done,
                traj_batch.action,
                traj_batch.value,
                traj_batch.reward,
                norm_ext_reward,
                norm_int_reward,
                traj_batch.log_prob,
                traj_batch.obs,
                traj_batch.norm_time_step,
                traj_batch.info,
            )

            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward, int_reward, norm_time_step, norm_ext_reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                    transition.int_reward,
                    transition.norm_time_step,
                    transition.norm_ext_reward,
                )
                rc_input = jnp.concatenate(
                    (norm_ext_reward[:, None], int_reward[:, None], norm_time_step[:, None]),
                    axis=-1,
                )
                int_lambda = rc_network.apply(rc_params, rc_input)
                delta = (
                    (reward + (int_reward * int_lambda))
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
                rnd_ext_return_norm_params,
            )

        (
            advantages,
            targets,
            norm_int_rewards,
            rnd_int_return_norm_params,
            rnd_ext_return_norm_params,
        ) = _calculate_gae(
            traj_batch, last_val, rnd_int_return_norm_params, rnd_ext_return_norm_params
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
                # (loss, vloss, aloss, entropy, rnd_loss, grads, rnd_grads) = jax.lax.pmean(
                #     (loss, vloss, aloss, entropy, rnd_loss, grads, rnd_grads), axis_name="devices"
                # )
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

            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            assert (
                batch_size == (config["NUM_STEPS"]) * config["NUM_ENVS_PER_DEVICE"]
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
            rnd_ext_return_norm_params,
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
        rnd_ext_return_norm_params,
        obs_norm_params,
        _rng,
    )
    runner_state, extra_info = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
    metric, int_rewards, norm_int_rewards, rl_total_loss = extra_info
    rewards = metric["sum_of_rewards"].mean(axis=-1)
    rewards = rewards.reshape(-1)
    rewards = rewards[-1]
    return {
        "rewards": rewards,
    }


for env_name in environments:
    rng = jax.random.PRNGKey(config["SEED"])
    config, env, env_params = make_config_env(config, env_name)
    print(f"Training in {config['ENV_NAME']}")

    make_train = jax.jit(jax.vmap(ppo_make_train, out_axes=(1, 0, 0, 0, 0)))
    train_fn = jax.vmap(train, in_axes=(0, None, 0, 0, 0, 0))
    train_fn = jax.vmap(train_fn, in_axes=(None, 0, None, None, None, None))
    train_fn = jax.pmap(train_fn, axis_name="devices")

    group = "reward_combiners"
    tags = ["meta-learner", config["ENV_NAME"]]
    name = f'{config["RUN_NAME"]}_{config["ENV_NAME"]}'
    fit_log = wandb.init(
        project="MetaLearnCuriosity",
        config=config,
        group=group,
        tags=tags,
        name=f"{name}_fitness",
    )
    reward_combiner_network = TemporalRewardCombiner()
    rc_params_pholder = reward_combiner_network.init(
        jax.random.PRNGKey(config["RC_SEED"]), jnp.zeros((1, 3))
    )
    es_rng = jax.random.PRNGKey(config["ES_SEED"])
    strategy = OpenES(
        popsize=config["POP_SIZE"],
        pholder_params=rc_params_pholder,
        opt_name="adam",
        lrate_decay=1,
        sigma_decay=0.999,
        sigma_init=0.04,
    )

    es_rng, es_rng_init = jax.random.split(es_rng)
    es_params = strategy.default_params
    es_state = strategy.initialize(es_rng_init, es_params)

    for _ in tqdm(range(config["NUM_GENERATIONS"]), desc="Processing Generations"):
        rng, rng_seeds = jax.random.split(rng)
        rng_train = jax.random.split(rng_seeds, config["NUM_SEEDS"])

        # setting up the RL agents.
        (
            rng_train,
            train_state,
            pred_state,
            target_params,
            init_rnd_obs,
        ) = make_train(rng_train)

        # duplicating here for pmap
        train_state = replicate(train_state, jax.local_devices())
        pred_state = replicate(pred_state, jax.local_devices())
        target_params = replicate(target_params, jax.local_devices())
        init_rnd_obs = replicate(init_rnd_obs, jax.local_devices())
        t = time.time()

        # Fitness evaulation
        es_rng, es_rng_ask = jax.random.split(es_rng)
        x, es_state = strategy.ask(es_rng_ask, es_state, es_params)
        output = jax.block_until_ready(
            train_fn(
                rng_train,
                x,
                train_state,
                pred_state,
                target_params,
                init_rnd_obs,
            )
        )
        fitness = output["rewards"].mean(-1)
        es_state = strategy.tell(x, fitness, es_state, es_params)
        elapsed_time = time.time() - t
        print(f"Done in {elapsed_time / 60:.2f}min")
        fit_log.log(
            {f"{name}_mean_fitness": fitness.mean(), f"{name}_best_fitness": jnp.max(fitness)}
        )

    fit_log.finish()
    logger = WBLogger(
        config=config,
        group=group,
        tags=tags,
        name=name,
    )
    # Get the absolute path of the directory
    checkpoint_directory = f'MLC_logs/flax_ckpt/{config["ENV_NAME"]}/{config["RUN_NAME"]}'
    params = strategy.param_reshaper.reshape_single(es_state.mean[0])
    path = os.path.abspath(checkpoint_directory)
    Save(path, params)
    logger.save_artifact(path)
    shutil.rmtree(path)
