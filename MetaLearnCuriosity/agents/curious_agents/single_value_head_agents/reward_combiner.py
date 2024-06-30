import os
from typing import Any, NamedTuple, Sequence

# from evosax import ParameterReshaper
import distrax
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
from MetaLearnCuriosity.agents.nn import TemporalRewardCombiner
from MetaLearnCuriosity.checkpoints import Save
from MetaLearnCuriosity.logger import WBLogger
from MetaLearnCuriosity.pmapped_open_es import OpenES
from MetaLearnCuriosity.utils import lifetime_return
from MetaLearnCuriosity.wrappers import FlattenObservationWrapper, LogWrapper, VecEnv

config = {
    "RUN_NAME": "temp_rc_lifetime_return",
    "PPO_SEED": 42,
    "RC_SEED": 43,
    "ES_SEED": 69,
    "NUM_SEEDS": 2,
    "LR": 2.5e-4,
    "NUM_ENVS": 16,
    "NUM_STEPS": 64,
    "TOTAL_TIMESTEPS": 5e5,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ENV_NAME": "Empty-misc",
    "ANNEAL_LR": True,
    "DEBUG": False,
    "EMA_PARAMETER": 0.99,
    "REW_NORM_PARAMETER": 0.99,
    "LIFETIME_GAMMA": 1,
    "POP_SIZE": 64,
    "NUM_GENERATIONS": 10,
}
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
config["TRAINING_HORIZON"] = config["TOTAL_TIMESTEPS_PER_DEVICE"] // config["NUM_ENVS_PER_DEVICE"]
env, env_params = gymnax.make(config["ENV_NAME"])
env = FlattenObservationWrapper(env)
env = LogWrapper(env)
env = VecEnv(env)
# THE NETWORKS


class TargetEncoder(nn.Module):
    encoder_layer_out_shape: Sequence[int]

    @nn.compact
    def __call__(self, x):
        encoded_obs = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        encoded_obs = nn.relu(encoded_obs)
        encoded_obs = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(1.0)),
            bias_init=constant(0.0),
        )(encoded_obs)

        return encoded_obs


class OnlineEncoder(nn.Module):
    encoder_layer_out_shape: Sequence[int]

    @nn.compact
    def __call__(self, x):
        encoded_obs = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        encoded_obs = nn.relu(encoded_obs)
        encoded_obs = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(1.0)),
            bias_init=constant(0.0),
        )(encoded_obs)

        return encoded_obs


# The predictor
class OnlinePredictor(nn.Module):
    encoder_layer_out_shape: Sequence[int]

    @nn.compact
    def __call__(self, x):

        # ! concatenation of action and obs does not work in the call function

        layer_out = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        layer_out = nn.tanh(layer_out)
        layer_out = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(layer_out)
        layer_out = nn.tanh(layer_out)
        layer_out = nn.Dense(
            self.encoder_layer_out_shape,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(layer_out)
        return layer_out


class BYOLActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # THE ACTOR MEAN
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # THE CRITIC
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


# THE TRANSITION CLASS
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    int_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    norm_time_step: jnp.ndarray
    info: jnp.ndarray


def byol_make_train(rng):  # noqa: C901
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # INIT NETWORK
    action_dim = env.action_space(env_params).n
    encoder_layer_out_shape = 64
    network = BYOLActorCritic(env.action_space(env_params).n, activation=config["ACTIVATION"])
    predicator = OnlinePredictor(encoder_layer_out_shape)
    target = TargetEncoder(encoder_layer_out_shape)
    online = OnlineEncoder(encoder_layer_out_shape)

    rng, _target_rng = jax.random.split(rng)
    rng, _online_rng = jax.random.split(rng)
    rng, _predicator_rng = jax.random.split(rng)
    rng, _network_rng = jax.random.split(rng)

    init_x = jnp.zeros(env.observation_space(env_params).shape)

    encoded_x = jnp.zeros((env.observation_space(env_params).shape[0], encoder_layer_out_shape))
    one_hot_encoded = jnp.zeros(
        (
            env.observation_space(env_params).shape[0],
            (encoder_layer_out_shape + action_dim),
        )
    )

    target_params = target.init(_target_rng, init_x)
    online_params = online.init(_online_rng, init_x)
    predicator_params = predicator.init(_predicator_rng, one_hot_encoded)
    network_params = network.init(_network_rng, encoded_x)

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
    network_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    byol_tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    online_state = TrainState.create(apply_fn=online.apply, params=online_params, tx=byol_tx)
    predicator_state = TrainState.create(
        apply_fn=predicator.apply, params=predicator_params, tx=byol_tx
    )
    rng = jax.random.split(rng, jax.local_device_count())
    return rng, network_state, online_state, predicator_state, target_params


def train(rng, rc_params, network_state, online_state, predicator_state, target_params):
    reward_combiner_network = TemporalRewardCombiner()
    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS_PER_DEVICE"])
    obsv, env_state = env.reset(reset_rng, env_params)
    target = TargetEncoder(64)
    action_dim = env.action_space(env_params).n

    # TRAIN LOOP

    def _update_step(runner_state, unused):
        # COLLECT TRAJECTORIES
        def _env_step(runner_state, unused):
            (
                network_state,
                online_state,
                predicator_state,
                target_params,
                env_state,
                last_obs,
                r_bar,
                r_bar_sq,
                c,
                mu_l,
                rng,
            ) = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            encoded_last_obs = online_state.apply_fn(online_state.params, last_obs)
            pi, value = network_state.apply_fn(network_state.params, encoded_last_obs)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS_PER_DEVICE"])
            obsv, env_state, reward, time_step, done, info = env.step(
                rng_step, env_state, action, env_params
            )
            # WORLD MODEL PREDICATION AND TARGET PREDICATION
            one_hot_action = jax.nn.one_hot(action, action_dim)
            encoded_one_hot = jnp.concatenate((encoded_last_obs, one_hot_action), axis=-1)
            pred_obs = predicator_state.apply_fn(predicator_state.params, encoded_one_hot)
            tar_obs = target.apply(target_params, obsv)

            # int reward
            pred_norm = (pred_obs) / (jnp.linalg.norm(pred_obs, axis=1)[:, None])
            tar_norm = jax.lax.stop_gradient(
                (tar_obs) / (jnp.linalg.norm(tar_obs, axis=1)[:, None])
            )
            int_reward = jnp.square(jnp.linalg.norm((pred_norm - tar_norm), axis=1)) * (1 - done)

            # normalise time step
            norm_time_step = time_step / config["TRAINING_HORIZON"]

            transition = Transition(
                done,
                action,
                value,
                reward,
                int_reward,
                log_prob,
                last_obs,
                obsv,
                norm_time_step,
                info,
            )

            runner_state = (
                network_state,
                online_state,
                predicator_state,
                target_params,
                env_state,
                obsv,
                r_bar,
                r_bar_sq,
                c,
                mu_l,
                rng,
            )

            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

        # ** Uncertainty calculation is not needed here.
        # ** The uncertainty is just given by int_reward.
        # ** This is because K=1, the open loop horizon is 1.

        # CALCULATE ADVANTAGE
        (
            network_state,
            online_state,
            predicator_state,
            target_params,
            env_state,
            last_obs,
            r_bar,
            r_bar_sq,
            c,
            mu_l,
            rng,
        ) = runner_state
        encoded_last_obs = online_state.apply_fn(online_state.params, last_obs)
        _, last_val = network_state.apply_fn(network_state.params, encoded_last_obs)

        def _update_reward_prior_norm(norm_int_reward, mu_l):
            mu_l = (
                config["REW_NORM_PARAMETER"] * mu_l
                + (1 - config["REW_NORM_PARAMETER"]) * norm_int_reward.mean()
            )
            return mu_l

        def _update_reward_norm_params(r_bar, r_bar_sq, c, r_c, r_c_sq):
            r_bar = config["REW_NORM_PARAMETER"] * r_bar + (1 - config["REW_NORM_PARAMETER"]) * r_c
            r_bar_sq = (
                config["REW_NORM_PARAMETER"] * r_bar_sq
                + (1 - config["REW_NORM_PARAMETER"]) * r_c_sq
            )
            c = c + 1

            return r_bar, r_bar_sq, c

        def _normlise_prior_int_rewards(int_reward, r_bar, r_bar_sq, c, mu_l):

            r_c = int_reward.mean()
            r_c_sq = jnp.square(int_reward).mean()
            r_bar, r_bar_sq, c = _update_reward_norm_params(r_bar, r_bar_sq, c, r_c, r_c_sq)
            mu_r = r_bar / (1 - config["REW_NORM_PARAMETER"] ** c)
            mu_r_sq = r_bar_sq / (1 - config["REW_NORM_PARAMETER"] ** c)
            mu_array = jnp.array([0, mu_r_sq - jnp.square(mu_r)])
            sigma_r = jnp.sqrt(jnp.max(mu_array) + 10e-8)
            norm_int_reward = int_reward / sigma_r
            mu_l = _update_reward_prior_norm(norm_int_reward, mu_l)
            prior_norm_int_reward = jnp.maximum(norm_int_reward - mu_l, 0)
            return prior_norm_int_reward, r_bar, r_bar_sq, c, mu_l

        def _calculate_gae(traj_batch, last_val, r_bar, r_bar_sq, c, mu_l):
            prior_norm_int_reward, r_bar, r_bar_sq, c, mu_l = _normlise_prior_int_rewards(
                traj_batch.int_reward, r_bar, r_bar_sq, c, mu_l
            )
            # * I want to loop over the Transitions which is why I am making a new Transition object
            norm_traj_batch = Transition(
                traj_batch.done,
                traj_batch.action,
                traj_batch.value,
                traj_batch.reward,
                prior_norm_int_reward,
                traj_batch.log_prob,
                traj_batch.obs,
                traj_batch.next_obs,
                traj_batch.norm_time_step,
                traj_batch.info,
            )

            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward, int_reward, norm_time_step = (
                    transition.done,
                    transition.value,
                    transition.reward,
                    transition.int_reward,
                    transition.norm_time_step,
                )
                rc_input = jnp.concatenate(
                    (reward[:, None], int_reward[:, None], norm_time_step[:, None]), axis=-1
                )
                int_lambda = reward_combiner_network.apply(rc_params, rc_input)
                total_reward = reward + (int_lambda * int_reward)
                delta = total_reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            # Looping over time steps in the "batch".
            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                norm_traj_batch,
                reverse=True,
                unroll=16,
            )
            return (
                advantages,
                advantages + norm_traj_batch.value,
                r_bar,
                r_bar_sq,
                c,
                mu_l,
            )

        advantages, targets, r_bar, r_bar_sq, c, mu_l = _calculate_gae(
            traj_batch, last_val, r_bar, r_bar_sq, c, mu_l
        )

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_states, batch_info):
                (
                    network_state,
                    online_state,
                    predicator_state,
                    target_params,
                ) = train_states
                traj_batch, advantages, targets = batch_info

                def _byol_loss(predicator_params, online_params, target_params, traj_batch):
                    encoded_obs = online_state.apply_fn(online_params, traj_batch.obs)
                    one_hot_action = jax.nn.one_hot(traj_batch.action, action_dim)
                    encoded_one_hot = jnp.concatenate((encoded_obs, one_hot_action), axis=-1)

                    pred_obs = predicator_state.apply_fn(
                        predicator_params,
                        encoded_one_hot,
                    )
                    tar_obs = target.apply(target_params, traj_batch.next_obs)
                    pred_norm = (pred_obs) / (jnp.linalg.norm(pred_obs, axis=1)[:, None])
                    tar_norm = jax.lax.stop_gradient(
                        (tar_obs) / (jnp.linalg.norm(tar_obs, axis=1)[:, None])
                    )
                    loss = jnp.square(jnp.linalg.norm((pred_norm - tar_norm), axis=1)) * (
                        1 - traj_batch.done
                    )
                    return loss.mean()

                def _rl_loss_fn(network_params, online_params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    encoded_obs = online_state.apply_fn(online_params, traj_batch.obs)
                    pi, value = network_state.apply_fn(network_params, encoded_obs)
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

                    rl_total_loss = (
                        loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                    )
                    return rl_total_loss, (value_loss, loss_actor, entropy)

                def _encoder_loss(
                    network_params,
                    predicator_params,
                    online_params,
                    target_params,
                    traj_batch,
                    gae,
                    targets,
                ):
                    rl_loss, _ = _rl_loss_fn(
                        network_params,
                        online_params,
                        traj_batch,
                        gae,
                        targets,
                    )

                    byol_loss = _byol_loss(
                        predicator_params,
                        online_params,
                        target_params,
                        traj_batch,
                    )

                    encoder_total_loss = rl_loss + byol_loss

                    return encoder_total_loss

                rl_grad_fn = jax.value_and_grad(_rl_loss_fn, has_aux=True)
                (rl_total_loss, (vloss, aloss, eloss)), rl_grads = rl_grad_fn(
                    network_state.params,
                    online_state.params,
                    traj_batch,
                    advantages,
                    targets,
                )

                byol_grad_fn = jax.value_and_grad(_byol_loss)
                byol_loss, byol_grad = byol_grad_fn(
                    predicator_state.params,
                    online_state.params,
                    target_params,
                    traj_batch,
                )

                encoder_grad_fn = jax.value_and_grad(_encoder_loss, 2)
                encoder_loss, encoder_grad = encoder_grad_fn(
                    network_state.params,
                    predicator_state.params,
                    online_state.params,
                    target_params,
                    traj_batch,
                    advantages,
                    targets,
                )
                # (rl_total_loss,vloss,aloss,eloss,byol_loss, encoder_loss,rl_grads,byol_grad,encoder_grad) = jax.lax.pmean((rl_total_loss,vloss,aloss,eloss,byol_loss, encoder_loss,rl_grads,byol_grad,encoder_grad), axis_name="devices")

                network_state = network_state.apply_gradients(grads=rl_grads)
                online_state = online_state.apply_gradients(grads=encoder_grad)
                predicator_state = predicator_state.apply_gradients(grads=byol_grad)
                target_params = jax.tree_util.tree_map(
                    lambda target_params, online_params: target_params * config["EMA_PARAMETER"]
                    + (1 - config["EMA_PARAMETER"]) * online_params,
                    target_params,
                    online_state.params,
                )

                return (
                    network_state,
                    online_state,
                    predicator_state,
                    target_params,
                ), ((rl_total_loss, (vloss, aloss, eloss)), byol_loss, encoder_loss)

            train_states, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            assert (
                batch_size == (config["NUM_STEPS"]) * config["NUM_ENVS_PER_DEVICE"]
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                shuffled_batch,
            )
            # ** We are looping over the minibatches
            train_states, total_losses = jax.lax.scan(_update_minbatch, train_states, minibatches)
            update_state = (train_states, traj_batch, advantages, targets, rng)
            return update_state, total_losses

        train_states = (
            network_state,
            online_state,
            predicator_state,
            target_params,
        )
        update_state = (train_states, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )
        train_states = update_state[0]
        network_state, online_state, predicator_state, target_params = train_states
        metric = traj_batch.info
        int_reward = traj_batch.int_reward
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
            network_state,
            online_state,
            predicator_state,
            target_params,
            env_state,
            last_obs,
            r_bar,
            r_bar_sq,
            c,
            mu_l,
            rng,
        )
        return runner_state, (metric, loss_info, int_reward, traj_batch)

    rng, _rng = jax.random.split(rng)
    r_bar = 0
    r_bar_sq = 0
    mu_l = 0
    c = 1
    runner_state = (
        network_state,
        online_state,
        predicator_state,
        target_params,
        env_state,
        obsv,
        r_bar,
        r_bar_sq,
        c,
        mu_l,
        _rng,
    )
    runner_state, extra_info = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
    metric, loss_info, int_reward, traj_batch = extra_info
    rl_total_loss, byol_loss, encoder_loss = loss_info
    return {
        "runner_state": runner_state,
        "metrics": metric,
        "int_reward": int_reward,
        "rl_total_loss": rl_total_loss[0],
        "rl_value_loss": rl_total_loss[1][0],
        "rl_actor_loss": rl_total_loss[1][1],
        "rl_entrophy_loss": rl_total_loss[1][2],
        "byol_loss": byol_loss,
        "encoder_loss": encoder_loss,
        "traj_batch": traj_batch,
    }


# Set up fitness function

train_fn = jax.vmap(train, in_axes=(0, None, 0, 0, 0, 0))
train_fn = jax.vmap(train_fn, in_axes=(None, 0, None, None, None, None))
train_fn = jax.pmap(train_fn, axis_name="devices")
v_make_train = jax.jit(jax.vmap(byol_make_train, out_axes=(1, 0, 0, 0, 0)))
devices = jax.local_devices()

reward_combiner_network = TemporalRewardCombiner()
rc_params_pholder = reward_combiner_network.init(
    jax.random.PRNGKey(config["RC_SEED"]), jnp.zeros((1, 3))
)
es_rng = jax.random.PRNGKey(config["ES_SEED"])
strategy = OpenES(
    popsize=config["POP_SIZE"],
    pholder_params=rc_params_pholder,
    opt_name="adam",
    lrate_init=2e-4,
)

es_rng, es_rng_init = jax.random.split(es_rng, 2)
es_params = strategy.default_params
es_state = strategy.initialize(es_rng_init, es_params)
ppo_rng = jax.random.PRNGKey(config["PPO_SEED"])


def es_step(
    es_rng, es_state, train_rng, network_state, online_state, predicator_state, target_params
):

    es_rng, es_rng_ask = jax.random.split(es_rng)
    x, es_state = strategy.ask(es_rng_ask, es_state, es_params)

    output = train_fn(train_rng, x, network_state, online_state, predicator_state, target_params)
    rewards = output["traj_batch"].reward
    fitness = rewards.mean(-1).mean(2).reshape(rewards.shape[0], rewards.shape[1], -1).sum(-1)
    es_state = strategy.tell(x, fitness, es_state, es_params)
    return es_rng, es_state, fitness


fitnesses = []

# Use tqdm to create a progress bar
for _ in tqdm(range(config["NUM_GENERATIONS"]), desc="Processing Generations"):
    ppo_rng, num_seeds_rng = jax.random.split(ppo_rng)
    num_seeds_rng = jax.random.split(num_seeds_rng, config["NUM_SEEDS"])
    train_rng, network_state, online_state, predicator_state, target_params = v_make_train(
        num_seeds_rng
    )

    # Replicate states across devices
    network_state = replicate(network_state, devices)
    online_state = replicate(online_state, devices)
    predicator_state = replicate(predicator_state, devices)
    target_params = replicate(target_params, devices)

    # Perform es_step and collect fitness
    es_rng, es_state, fitness = es_step(
        es_rng, es_state, train_rng, network_state, online_state, predicator_state, target_params
    )
    fitnesses.append(fitness)
