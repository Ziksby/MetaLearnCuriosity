from typing import Any, NamedTuple, Sequence

import distrax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from MetaLearnCuriosity.checkpoints import Save
from MetaLearnCuriosity.logger import WBLogger
from MetaLearnCuriosity.utils import make_obs_gymnax_discrete
from MetaLearnCuriosity.wrappers import FlattenObservationWrapper, LogWrapper

# * Consider using RELU activation function.


class TargetNetwork(nn.Module):
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


class PredictorNetwork(nn.Module):
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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    int_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def ppo_make_train(config):  # noqa: C901
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def update_obs_norm_params(batch_mean, batch_var, count, obs_mean, obs_var):
        batch_count = config["NUM_ENVS"]
        tot_count = count + batch_count
        delta = batch_mean - obs_mean

        new_obs_mean = obs_mean + delta * batch_count / tot_count
        m_a = obs_var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
        new_obs_var = M2 / tot_count
        new_count = tot_count

        return new_count, new_obs_mean, new_obs_var

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = PPOActorCritic(env.action_space(env_params).n, activation=config["ACTIVATION"])
        encoder_layer_out_shape = 64
        predictor = PredictorNetwork(encoder_layer_out_shape)
        target = TargetNetwork(encoder_layer_out_shape)

        rng, _rng = jax.random.split(rng)
        rng, _pred_rng = jax.random.split(rng)
        rng, _tar_rng = jax.random.split(rng)

        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        target_params = target.init(_tar_rng, init_x)
        pred_params = predictor.init(_pred_rng, init_x)

        # # OBS NORM PARAMS
        # * Coming soon to a pull request near you!
        random_rollout = jax.jit(make_obs_gymnax_discrete(config, env, env_params))
        init_obs = jnp.transpose(random_rollout(rng), (1, 0, 2))

        init_mean_obs = jnp.zeros(init_obs.shape[-1])
        init_var_obs = jnp.ones(init_obs.shape[-1])
        init_obs_count = 1e-4

        obs_batch_mean = init_obs.mean((0, 1))
        obs_batch_var = jnp.var(init_obs, (0, 1))

        obs_count, obs_mean, obs_var = update_obs_norm_params(
            obs_batch_mean, obs_batch_var, init_obs_count, init_mean_obs, init_var_obs
        )

        # INT REWARD NORM PARAMS
        rewems = jnp.zeros(config["NUM_STEPS"], dtype=jnp.float32)
        count = 1e-4
        int_mean = 0.0
        int_var = 1.0

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
        pred_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
        network_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        predictor_state = TrainState.create(
            apply_fn=predictor.apply,
            params=pred_params,
            tx=pred_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    network_state,
                    predictor_state,
                    target_params,
                    env_state,
                    last_obs,
                    count,
                    rewems,
                    int_mean,
                    int_var,
                    obs_count,
                    obs_mean,
                    obs_var,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(network_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )

                # INT REWARD
                norm_obsv = ((obsv - obs_mean[None, :]) / (jnp.sqrt(obs_var[None, :]) + 1e-8)).clip(
                    -5, 5
                )

                tar_obsv = jax.lax.stop_gradient(target.apply(target_params, norm_obsv))
                pred_obsv = predictor.apply(predictor_state.params, norm_obsv)
                int_reward = jnp.square(jnp.linalg.norm((pred_obsv - tar_obsv), axis=1))

                transition = Transition(
                    done, action, value, reward, int_reward, log_prob, last_obs, info
                )
                runner_state = (
                    network_state,
                    predictor_state,
                    target_params,
                    env_state,
                    obsv,
                    count,
                    rewems,
                    int_mean,
                    int_var,
                    obs_count,
                    obs_mean,
                    obs_var,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                network_state,
                predictor_state,
                target_params,
                env_state,
                last_obs,
                count,
                rewems,
                int_mean,
                int_var,
                obs_count,
                obs_mean,
                obs_var,
                rng,
            ) = runner_state
            _, last_val = network.apply(network_state.params, last_obs)

            def _update_int_reward_norm_params(batch_mean, batch_var, count, int_mean, int_var):

                batch_count = config["NUM_ENVS"]
                tot_count = count + batch_count
                delta = batch_mean - int_mean

                new_int_mean = int_mean + delta * batch_count / tot_count
                m_a = int_var * count
                m_b = batch_var * batch_count
                M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
                new_int_var = M2 / tot_count
                new_count = tot_count

                return new_count, new_int_mean, new_int_var

            def _normalise_int_rewards(traj_batch, count, rewems, int_mean, int_var):
                def _multiply_rewems_w_dones(rewems, dones_row):
                    rewems = rewems * (1 - dones_row)
                    return rewems, rewems

                def _update_rewems(rewems, int_reward_row):
                    rewems = rewems * config["INT_GAMMA"] + int_reward_row
                    return rewems, rewems

                int_reward = traj_batch.int_reward

                int_reward_transpose = jnp.transpose(int_reward)

                rewems, _ = jax.lax.scan(
                    _multiply_rewems_w_dones, rewems, jnp.transpose(traj_batch.done)
                )

                rewems, dis_int_reward_transpose = jax.lax.scan(
                    _update_rewems, rewems, int_reward_transpose
                )

                batch_mean = dis_int_reward_transpose.mean()
                batch_var = jnp.var(dis_int_reward_transpose)

                count, int_mean, int_var = _update_int_reward_norm_params(
                    batch_mean, batch_var, count, int_mean, int_var
                )
                norm_int_reward = int_reward / jnp.sqrt(int_var + 1e-8)
                return norm_int_reward, count, rewems, int_mean, int_var

            def _calculate_gae(traj_batch, last_val, count, rewems, int_mean, int_var):
                norm_int_reward, count, rewems, int_mean, int_var = _normalise_int_rewards(
                    traj_batch, count, rewems, int_mean, int_var
                )
                # * I want to loop over the Transitions which is why I am making a new Transition object
                norm_traj_batch = Transition(
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
                        (reward + config["INT_LAMBDA"] * int_reward)
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
                return advantages, advantages + traj_batch.value, count, rewems, int_mean, int_var

            advantages, targets, count, rewems, int_mean, int_var = _calculate_gae(
                traj_batch, last_val, count, rewems, int_mean, int_var
            )

            obs_batch_mean = traj_batch.obs.mean((0, 1))
            obs_batch_var = jnp.var(traj_batch.obs, (0, 1))
            obs_count, obs_mean, obs_var = update_obs_norm_params(
                obs_batch_mean, obs_batch_var, obs_count, obs_mean, obs_var
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    network_state, predictor_state, target_params = train_states
                    traj_batch, advantages, targets, rnd_obs = batch_info

                    def _rnd_loss(pred_params, target_params, rnd_obs):
                        # RERUN NETWORK

                        tar_obs = jax.lax.stop_gradient(target.apply(target_params, rnd_obs))
                        pred_obs = predictor.apply(pred_params, rnd_obs)
                        loss = jnp.square(jnp.linalg.norm((pred_obs - tar_obs), axis=1))
                        return loss.mean()

                    def _rl_loss_fn(network_params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(network_params, traj_batch.obs)
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

                    grad_rl_fn = jax.value_and_grad(_rl_loss_fn, has_aux=True)
                    grad_rnd_fn = jax.value_and_grad(_rnd_loss)

                    total_rl_loss, rl_grads = grad_rl_fn(
                        network_state.params, traj_batch, advantages, targets
                    )

                    rnd_loss, rnd_grad = grad_rnd_fn(predictor_state.params, target_params, rnd_obs)
                    network_state = network_state.apply_gradients(grads=rl_grads)
                    predictor_state = predictor_state.apply_gradients(grads=rnd_grad)
                    train_states = network_state, predictor_state, target_params
                    return train_states, (total_rl_loss, rnd_loss)

                (
                    train_states,
                    traj_batch,
                    advantages,
                    targets,
                    obs_count,
                    obs_mean,
                    obs_var,
                    rng,
                ) = update_state

                # Norm the observation

                rnd_obs = (
                    (traj_batch.obs - obs_mean[None, :]) / (jnp.sqrt(obs_var[None, :]) + 10e-8)
                ).clip(-5, 5)

                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets, rnd_obs)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_states, total_loss = jax.lax.scan(_update_minbatch, train_states, minibatches)
                update_state = (
                    train_states,
                    traj_batch,
                    advantages,
                    targets,
                    obs_count,
                    obs_mean,
                    obs_var,
                    rng,
                )
                return update_state, total_loss

            train_states = network_state, predictor_state, target_params
            update_state = (
                train_states,
                traj_batch,
                advantages,
                targets,
                obs_count,
                obs_mean,
                obs_var,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_states = update_state[0]
            int_reward = traj_batch.int_reward
            network_state, predictor_state, target_params = train_states
            metric = traj_batch.info
            obs_count = update_state[-4]
            obs_mean = update_state[-3]
            obs_var = update_state[-2]
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")

                jax.debug.callback(callback, metric)

            runner_state = (
                network_state,
                predictor_state,
                target_params,
                env_state,
                last_obs,
                count,
                rewems,
                int_mean,
                int_var,
                obs_count,
                obs_mean,
                obs_var,
                rng,
            )
            return runner_state, (metric, loss_info, int_reward)

        rng, _rng = jax.random.split(rng)
        runner_state = (
            network_state,
            predictor_state,
            target_params,
            env_state,
            obsv,
            count,
            rewems,
            int_mean,
            int_var,
            obs_count,
            obs_mean,
            obs_var,
            _rng,
        )
        runner_state, extra_info = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        metric, total_loss, int_reward = extra_info
        rl_total_loss, rnd_loss = total_loss
        return {
            "runner_state": runner_state,
            "metrics": metric,
            "int_reward": int_reward,
            "rl_total_loss": rl_total_loss[0],
            "rl_value_loss": rl_total_loss[1][0],
            "rl_actor_loss": rl_total_loss[1][1],
            "rl_entrophy_loss": rl_total_loss[1][2],
            "rnd_loss": rnd_loss,
        }

    return train


if __name__ == "__main__":
    config = {
        "RUN_NAME": "rnd_empty",
        "SEED": 42,
        "NUM_SEEDS": 30,
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
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
        "INT_GAMMA": 0.999,
        "INT_LAMBDA": 0.08,
    }

    rng = jax.random.PRNGKey(config["SEED"])

    if config["NUM_SEEDS"] > 1:
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(ppo_make_train(config)))
        output = train_vjit(rngs)

    else:
        train_jit = jax.jit(ppo_make_train(config))
        output = train_jit(rng)

    logger = WBLogger(
        config=config,
        group=f"rnd/{config['ENV_NAME']}",
        tags=["rnd", config["ENV_NAME"]],
        name=config["RUN_NAME"],
    )
    logger.log_episode_return(output, config["NUM_SEEDS"])
    logger.log_rl_losses(output, config["NUM_SEEDS"])
    logger.log_int_rewards(output, config["NUM_SEEDS"])
    logger.log_rnd_losses(output, config["NUM_SEEDS"])
    output["config"] = config
    path = f'MLC_logs/flax_ckpt/{config["ENV_NAME"]}/{config["RUN_NAME"]}_{config["NUM_SEEDS"]}'
    Save(path, output)
