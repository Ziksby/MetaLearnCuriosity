import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
from MetaLearnCuriosity.wrappers import LogWrapper, FlattenObservationWrapper
import jax.tree_util

# THE NETWORKS


class TargetEncoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.tanh(actor_mean)
        return actor_mean


class OnlineEncoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.tanh(actor_mean)
        return actor_mean


# The predictor
class OnlinePredictor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x, action):

        # One-hot encode the action
        one_hot_action = jax.nn.one_hot(action, self.action_dim)

        inp = jnp.concatenate([x, one_hot_action], axis=-1)

        layer_out = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(inp)
        layer_out = nn.tanh(layer_out)
        layer_out = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(layer_out)
        layer_out = nn.tanh(layer_out)
        layer_out = nn.Dense(64, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            layer_out
        )
        return layer_out


class ActorCritic(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # THE CRITIC
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = nn.tanh(actor_mean)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)

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
    last_obs: jnp.ndarray
    info: jnp.ndarray


# HELPER FUNCTIONS
def l2_norm_squared(arr, axis=None):
    return jnp.sum(jnp.square(arr), axis=axis)


def l2_norm(arr):
    return jnp.sqrt(jnp.sum(jnp.square(arr)))


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(env.action_space(env_params).n)
        predicator = OnlinePredictor(env.action_space(env_params).n)
        target = TargetEncoder()
        online = OnlineEncoder()

        rng, _target_rng = jax.random.split(rng)
        rng, _online_rng = jax.random.split(rng)
        rng, _predicator_rng = jax.random.split(rng)
        rng, _network_rng = jax.random.split(rng)

        init_x = jnp.zeros(env.observation_space(env_params).shape)
        encoded_x = jnp.zeros((init_x.shape[0], 64))
        init_action = jnp.zeros(env.action_space(env_params).shape, dtype=jnp.int32)

        target_params = target.init(_target_rng, init_x)
        online_params = online.init(_online_rng, init_x)
        predicator_params = predicator.init(_predicator_rng, encoded_x, init_action)
        network_params = network.init(_network_rng, encoded_x)

        r_bar = 0
        r_bar_sq = 0
        c = 0
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

        online_state = TrainState.create(
            apply_fn=online.apply, params=online_params, tx=tx
        )
        predicator_state = TrainState.create(
            apply_fn=predicator.apply, params=predicator_params, tx=tx
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
                    online_state,
                    predicator_state,
                    target_params,
                    env_state,
                    last_obs,
                    rng,
                    r_bar,
                    r_bar_sq,
                    c,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                encoded_last_obs = online.apply(online_state.params, last_obs)
                pi, value = network.apply(network_state.params, encoded_last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)

                # WORLD MODEL PREDICATION AND TARGET PREDICATION
                pred_obs = predicator.apply(predicator_state.params, last_obs, action)
                tar_obs = target.apply(target_params, obsv)

                # int reward
                pred_norm = (pred_obs) / (l2_norm(pred_obs))
                tar_norm = (tar_obs) / (l2_norm(tar_obs))
                int_reward = l2_norm_squared(pred_norm - tar_norm) * (1 - done)

                transition = Transition(
                    done,
                    action,
                    value,
                    reward,
                    int_reward,
                    log_prob,
                    last_obs,
                    obsv,
                    info,
                )
                runner_state = (
                    network_state,
                    online_state,
                    predicator_state,
                    target_params,
                    env_state,
                    obsv,
                    rng,
                    r_bar,
                    r_bar_sq,
                    c,
                )

                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

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
                rng,
                r_bar,
                r_bar_sq,
                c,
            ) = runner_state
            encoded_last_obs = online.apply(online_state.params, last_obs)
            _, last_val = network.apply(network_state.params, encoded_last_obs)

            def _update_reward_norm_params(r_bar, r_bar_sq, c, r_c, r_c_sq):
                r_bar = (
                    config["REW_NORM_PARAMETER"] * r_bar
                    + (1 - config["REW_NORM_PARAMETER"]) * r_c
                )
                r_bar_sq = (
                    config["REW_NORM_PARAMETER"] * r_bar_sq
                    + (1 - config["REW_NORM_PARAMETER"]) * r_c_sq
                )
                c = c + 1

                return r_bar, r_bar_sq, c

            def _normlise_int_rewards(int_reward, r_bar, r_bar_sq, c):
                r_c = int_reward[:-1].mean()
                r_c_sq = jnp.square(int_reward[:-1]).mean()
                r_bar, r_bar_sq, c = _update_reward_norm_params(
                    r_bar, r_bar_sq, c, r_c, r_c_sq
                )
                mu_r = r_bar / (1 - config["REW_NORM_PARAMETER"])
                mu_r_sq = r_bar_sq / (1 - config["REW_NORM_PARAMETER"])
                mu_array = jnp.array([0, mu_r_sq - jnp.square(mu_r)])
                sigma_r = jnp.sqrt(jnp.max(mu_array) + 10e-8)
                norm_int_reward = int_reward[:-1] / sigma_r
                return norm_int_reward, r_bar, r_bar_sq, c

            def _calculate_gae(traj_batch, last_val, r_bar, r_bar_sq, c):
                norm_int_reward, r_bar, r_bar_sq, c = _normlise_int_rewards(
                    traj_batch.int_reward, r_bar, r_bar_sq, c
                )
                # ** I want to loop over the Transitions which is why I am making a new Transition object
                norm_traj_batch = Transition(
                    traj_batch.done,
                    traj_batch.action,
                    traj_batch.value,
                    traj_batch.reward,
                    norm_int_reward,
                    traj_batch.log_prob,
                    traj_batch.last_obs,
                    traj_batch.obsv,
                    traj_batch.info,
                )

                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )

                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    norm_traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value, r_bar, r_bar_sq, c

            advantages, targets, r_bar, r_bar_sq, c = _calculate_gae(
                traj_batch, last_val, r_bar, r_bar_sq, c
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

                    def _byol_loss(
                        predicator_params, online_params, target_params, traj_batch
                    ):
                        # encoded_obs = online.apply(online_params, traj_batch.obs)
                        encoded_last_obs = online.apply(
                            online_params, traj_batch.last_obs
                        )
                        pred_obs = predicator.apply(
                            predicator_params, encoded_last_obs, traj_batch.action
                        )
                        tar_obs = target.apply(target_params, traj_batch.obs)
                        pred_norm = (pred_obs) / (l2_norm(pred_obs))
                        tar_norm = (tar_obs) / (l2_norm(tar_obs))
                        loss = l2_norm_squared(pred_norm - tar_norm)
                        return loss.mean()

                    def _rl_loss_fn(
                        network_params, online_params, traj_batch, gae, targets
                    ):
                        # RERUN NETWORK
                        encoded_obs = online.apply(online_params, traj_batch.obs)
                        pi, value = network.apply(network_params, encoded_obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

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
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
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
                            network_state.params,
                            online_state.params,
                            traj_batch,
                            gae,
                            targets,
                        )

                        byol_loss = _byol_loss(
                            predicator_state.params,
                            online_state.params,
                            target_params,
                            traj_batch,
                        )

                        encoder_total_loss = rl_loss + byol_loss

                        # ? Should one take mean of the losses or just sum them together

                        return encoder_total_loss

                    rl_grad_fn = jax.value_and_grad(_rl_loss_fn, has_aux=True)
                    rl_total_loss, rl_grads = rl_grad_fn(
                        network_state.params, traj_batch, advantages, targets
                    )

                    byol_grad_fn = jax.grad(_byol_loss)
                    byol_grad = byol_grad_fn(
                        predicator_state.params,
                        online_state.params,
                        target_params,
                        traj_batch,
                    )

                    encoder_grad_fn = jax.grad(_encoder_loss, 2)
                    encoder_grad = encoder_grad_fn(
                        network_state.params,
                        predicator_state.params,
                        online_state.params,
                        target_params,
                        traj_batch,
                        advantages,
                        targets,
                    )

                    network_state = network_state.apply_gradients(grads=rl_grads)
                    online_state = online_state.apply_gradients(grads=encoder_grad)
                    predicator_state = predicator_state.apply_gradients(grads=byol_grad)
                    target_params = jax.tree_util.tree_map(
                        lambda target_params, online_params: target_params
                        * config["EMA_PARAMETER"]
                        + (1 - config["EMA_PARAMETER"]) * online_params,
                        target_params,
                        online_state.params,
                    )

                    return (
                        network_state,
                        online_state,
                        predicator_state,
                        target_params,
                    ), rl_total_loss

                train_states, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                # ** We are looping over the minibatches
                train_states, rl_total_loss = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (train_states, traj_batch, advantages, targets, rng)
                return update_state, rl_total_loss

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
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (
                network_state,
                online_state,
                predicator_state,
                target_params,
                env_state,
                last_obs,
                rng,
                r_bar,
                r_bar_sq,
                c,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            network_state,
            online_state,
            predicator_state,
            target_params,
            env_state,
            obsv,
            _rng,
            r_bar,
            r_bar_sq,
            c,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    config = {
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
        "ENV_NAME": "CartPole-v1",
        "ANNEAL_LR": True,
        "DEBUG": True,
        "EMA_PARAMETER": 0.99,
        "REW_NORM_PARAMETER": 0.99,
    }
    rng = jax.random.PRNGKey(42)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
    print("DONE!")
