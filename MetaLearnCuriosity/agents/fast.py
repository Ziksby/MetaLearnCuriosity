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
from MetaLearnCuriosity.wrappers import FlattenObservationWrapper, LogWrapper


class FAST(nn.Module):

    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        action = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        action = nn.relu(action)
        action = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(1.0)),
            bias_init=constant(0.0),
        )(action)
        action = nn.relu(action)
        action = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(
            action
        )
        pi = distrax.Categorical(logits=action)

        return pi


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
    norm_time_step: jnp.ndarray
    info: jnp.ndarray


def ppo_make_train(config):
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
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
        network = PPOActorCritic(env.action_space(env_params).n, activation=config["ACTIVATION"])
        fast = FAST(env.action_space(env_params).n)

        rng, _rng = jax.random.split(rng)
        rng, _fast_rng = jax.random.split(rng)

        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        fast_params = fast.init(_fast_rng, init_x)

        fast_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
        fast_state = TrainState.create(
            apply_fn=fast.apply,
            params=fast_params,
            tx=fast_tx,
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
        network_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
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
                    fast_state,
                    env_state,
                    last_obs,
                    count,
                    rewems,
                    int_mean,
                    int_var,
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
                # NORM TIME STEP
                norm_time_step = env_state.env_state.time / env_params.max_steps_in_episode
                # INTRINSIC REWARD
                fast_pi_st = fast.apply(fast_state.params, last_obs)
                fast_pi_st_1 = fast.apply(fast_state.params, obsv)
                int_reward = jnp.linalg.norm((fast_pi_st_1.logits - fast_pi_st.logits), axis=1)

                transition = Transition(
                    done,
                    action,
                    value,
                    reward,
                    int_reward,
                    log_prob,
                    last_obs,
                    norm_time_step,
                    info,
                )
                runner_state = (
                    network_state,
                    fast_state,
                    env_state,
                    obsv,
                    count,
                    rewems,
                    int_mean,
                    int_var,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                network_state,
                fast_state,
                env_state,
                last_obs,
                count,
                rewems,
                int_mean,
                int_var,
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

                # def _multiply_rewems_w_dones(rewems,dones_row):
                #     rewems = rewems * (1-dones_row)
                #     return rewems,rewems

                def _update_rewems(rewems, int_reward_row):
                    rewems = rewems * config["INT_GAMMA"] + int_reward_row
                    return rewems, rewems

                int_reward = traj_batch.int_reward

                int_reward_transpose = jnp.transpose(int_reward)

                # rewems,_ = jax.lax.scan(_multiply_rewems_w_dones,rewems,jnp.transpose(traj_batch.done))

                rewems, dis_int_reward_transpose = jax.lax.scan(
                    _update_rewems, rewems, int_reward_transpose
                )

                batch_mean = dis_int_reward_transpose.mean()
                batch_var = jnp.var(dis_int_reward_transpose)

                count, int_mean, int_var = _update_int_reward_norm_params(
                    batch_mean, batch_var, count, int_mean, int_var
                )
                norm_int_reward = int_reward / jnp.sqrt(int_var)
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
                    traj_batch.norm_time_step,
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
                    # total_reward = (1 + int_reward - norm_time_step) * int_reward + (
                    #     norm_time_step * reward
                    # )
                    total_reward = (
                        reward + config["INT_LAMBDA"] * int_reward
                    )  # total_reward / (1 + int_reward)
                    delta = total_reward + config["GAMMA"] * next_value * (1 - done) - value
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

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    traj_batch, advantages, targets = batch_info
                    network_state, fast_state = train_states

                    def _fast_loss_fn(fast_params, traj_batch):
                        # The NLL Loss
                        fast_pi = fast.apply(fast_params, traj_batch.obs)
                        return -fast_pi.log_prob(traj_batch.action).mean()

                    def _rl_loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
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

                    rl_grad_fn = jax.value_and_grad(_rl_loss_fn, has_aux=True)
                    fast_grad_fn = jax.value_and_grad(_fast_loss_fn)
                    total_loss, rl_grads = rl_grad_fn(
                        network_state.params, traj_batch, advantages, targets
                    )
                    fast_loss, fast_grads = fast_grad_fn(fast_state.params, traj_batch)
                    network_state = network_state.apply_gradients(grads=rl_grads)
                    fast_state = fast_state.apply_gradients(grads=fast_grads)
                    return (network_state, fast_state), (total_loss, fast_loss)

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
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_states, total_loss = jax.lax.scan(_update_minbatch, train_states, minibatches)
                update_state = (train_states, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            train_states = (network_state, fast_state)
            update_state = (train_states, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_states = update_state[0]
            network_state, fast_state = train_states
            metric = traj_batch.info
            int_reward = traj_batch.int_reward
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
                fast_state,
                env_state,
                last_obs,
                count,
                rewems,
                int_mean,
                int_var,
                rng,
            )
            return runner_state, (metric, loss_info, int_reward)

        rng, _rng = jax.random.split(rng)
        runner_state = (
            network_state,
            fast_state,
            env_state,
            obsv,
            count,
            rewems,
            int_mean,
            int_var,
            _rng,
        )
        runner_state, extra_info = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        metric, total_loss, int_reward = extra_info
        rl_total_loss, fast_loss = total_loss
        return {
            "runner_state": runner_state,
            "metrics": metric,
            "rl_total_loss": rl_total_loss[0],
            "rl_value_loss": rl_total_loss[1][0],
            "rl_actor_loss": rl_total_loss[1][1],
            "rl_entrophy_loss": rl_total_loss[1][2],
            "int_reward": int_reward,
            "fast_loss": fast_loss,
        }

    return train


if __name__ == "__main__":
    config = {
        "RUN_NAME": "fast",
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
        "INT_LAMBDA": 0.002,
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
        group=f"fast/{config['ENV_NAME']}",
        tags=["fast", config["ENV_NAME"]],
        name=config["RUN_NAME"],
    )
    logger.log_episode_return(output, config["NUM_SEEDS"])
    logger.log_rl_losses(output, config["NUM_SEEDS"])
    logger.log_fast_losses(output, config["NUM_SEEDS"])
    logger.log_int_rewards(output, config["NUM_SEEDS"])
    output["config"] = config
    path = f'MLC_logs/flax_ckpt/{config["ENV_NAME"]}/{config["RUN_NAME"]}_{config["NUM_SEEDS"]}'
    Save(path, output)
