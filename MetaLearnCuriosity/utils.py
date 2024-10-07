from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
from flax import struct
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState

from MetaLearnCuriosity.agents.nn import RewardCombiner, TargetNetwork


class RNDTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    int_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class RCRNDTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    norm_reward: jnp.ndarray
    int_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    norm_time_step: jnp.ndarray
    ext_reward_hist: jnp.ndarray
    int_reward_hist: jnp.ndarray
    info: jnp.ndarray


class RCBYOLTransition(NamedTuple):
    done: jnp.ndarray
    prev_action: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    norm_reward: jnp.ndarray  # will be normalised in batch
    int_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    bt: jnp.ndarray
    norm_time_step: jnp.ndarray
    ext_reward_hist: jnp.ndarray
    int_reward_hist: jnp.ndarray
    info: jnp.ndarray


class BYOLTransition(NamedTuple):
    done: jnp.ndarray
    prev_action: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    int_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    bt: jnp.ndarray
    info: jnp.ndarray


class BYOLLiteTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    int_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


class BYOLMiniGridTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    int_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    # for minigrid rnn policy
    prev_action: jnp.ndarray
    prev_reward: jnp.ndarray
    prev_bt: jnp.ndarray
    info: jnp.ndarray


class RCBYOLMiniGridTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    norm_reward: jnp.ndarray
    int_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    prev_action: jnp.ndarray
    prev_reward: jnp.ndarray
    prev_bt: jnp.ndarray
    norm_time_step: jnp.ndarray
    info: jnp.ndarray


class RNDMiniGridTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    int_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    # for minigrid rnn policy
    prev_action: jnp.ndarray
    prev_reward: jnp.ndarray
    info: jnp.ndarray


class BYOLRewardNorm(NamedTuple):
    ema_mean: float
    ema_mean_sq: float
    c: float
    mu_l: float


class MiniGridTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    # for minigrid rnn policy
    prev_action: jnp.ndarray
    prev_reward: jnp.ndarray
    info: jnp.ndarray


class PPOTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class RandomTransition(NamedTuple):
    obs: jnp.ndarray


class RandomAgentTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    info: jnp.ndarray


class ObsNormParams(NamedTuple):
    count: float
    mean: jnp.ndarray
    var: jnp.ndarray


class RNDNormIntReturnParams(NamedTuple):
    count: float
    mean: float
    var: float
    rewems: jnp.ndarray


def calculate_gae(
    transitions: MiniGridTransition,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    # single iteration for the loop
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        delta = transition.reward + gamma * next_value * (1 - transition.done) - transition.value
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.value


def rc_byol_calculate_gae(
    transitions: RCBYOLMiniGridTransition,
    rc_params,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
    rew_norm_parameter: float,
    byol_reward_norm_params: BYOLRewardNorm,
    ext_reward_norm_params: BYOLRewardNorm,
) -> tuple[jax.Array, jax.Array]:
    rc_network = RewardCombiner()
    norm_int_reward, byol_reward_norm_params = byol_normalize_prior_int_rewards(
        transitions.int_reward, byol_reward_norm_params, rew_norm_parameter
    )
    norm_ext_reward, ext_reward_norm_params = byol_normalize_prior_int_rewards(
        transitions.norm_reward, ext_reward_norm_params, rew_norm_parameter, prior=False
    )
    norm_traj_batch = RCBYOLMiniGridTransition(
        transitions.done,
        transitions.action,
        transitions.value,
        transitions.reward,
        norm_ext_reward,
        norm_int_reward,
        transitions.log_prob,
        transitions.obs,
        transitions.next_obs,
        # for minigrid rnn policy
        transitions.prev_action,
        transitions.prev_reward,
        transitions.prev_bt,
        transitions.norm_time_step,
        transitions.info,
    )
    # single iteration for the loop

    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        rc_input = jnp.concatenate(
            (transition.norm_reward[:, None], transition.int_reward[:, None]),
            axis=-1,
        )
        int_lambda = rc_network.apply(rc_params, rc_input)
        delta = (
            (transition.reward + (transition.int_reward * int_lambda))
            + gamma * next_value * (1 - transition.done)
            - transition.value
        )
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), (gae, int_lambda)

    _, (advantages, int_lambda) = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        norm_traj_batch,
        reverse=True,
    )
    # advantages and values (Q)
    return (
        advantages,
        advantages + transitions.value,
        norm_int_reward,
        norm_ext_reward,
        byol_reward_norm_params,
        ext_reward_norm_params,
        int_lambda,
    )


def byol_calculate_gae(
    transitions: BYOLMiniGridTransition,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
    int_lambda: float,
    rew_norm_parameter: float,
    byol_reward_norm_params: BYOLRewardNorm,
) -> tuple[jax.Array, jax.Array]:
    norm_int_reward, byol_reward_norm_params = byol_normalize_prior_int_rewards(
        transitions.int_reward, byol_reward_norm_params, rew_norm_parameter
    )
    norm_traj_batch = BYOLMiniGridTransition(
        transitions.done,
        transitions.action,
        transitions.value,
        transitions.reward,
        norm_int_reward,
        transitions.log_prob,
        transitions.obs,
        transitions.next_obs,
        # for minigrid rnn policy
        transitions.prev_action,
        transitions.prev_reward,
        transitions.prev_bt,
        transitions.info,
    )
    # single iteration for the loop

    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        delta = (
            (transition.reward + (transition.int_reward * int_lambda))
            + gamma * next_value * (1 - transition.done)
            - transition.value
        )
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        norm_traj_batch,
        reverse=True,
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.value, norm_int_reward, byol_reward_norm_params


def rnd_calculate_gae(
    transitions: RNDMiniGridTransition,
    last_val: jax.Array,
    gamma: float,
    int_gamma: float,
    gae_lambda: float,
    int_lambda: float,
    rnd_int_return_norm_params: RNDNormIntReturnParams,
) -> tuple[jax.Array, jax.Array]:
    # single iteration for the loop

    norm_int_reward, rnd_int_return_norm_params = rnd_normalise_int_rewards(
        transitions, rnd_int_return_norm_params, int_gamma
    )

    # *** Wrong transition class used here.
    # *** But it still works.
    norm_traj_batch = RNDTransition(
        transitions.done,
        transitions.action,
        transitions.value,
        transitions.reward,
        norm_int_reward,
        transitions.log_prob,
        transitions.obs,
        transitions.info,
    )

    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        delta = (
            (transition.reward + (int_lambda * transition.int_reward))
            + gamma * next_value * (1 - transition.done)
            - transition.value
        )
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        norm_traj_batch,
        reverse=True,
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.value, rnd_int_return_norm_params, norm_int_reward


# Get dummy data for Obs Norm
def make_obs_gymnax_discrete(num_envs, env, env_params, num_steps):
    def random_rollout(rng, env_params=env_params):
        """Rollout a jitted gymnax episode with lax.scan."""
        # Reset the environment
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs)
        obsv, env_state = env.reset(reset_rng, env_params)

        def step(runner_state, tmp):

            env_state, last_obs, rng = runner_state

            # SELECT ACTION
            rng, _rng, act_rngs = jax.random.split(rng, 3)
            act_rngs = jax.random.split(act_rngs, num_envs)
            action = jax.vmap(env.action_space(env_params).sample)(act_rngs)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, num_envs)
            obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
            transition = RandomTransition(last_obs)
            runner_state = (env_state, obsv, rng)
            return runner_state, transition

        runner_state = (env_state, obsv, rng)

        runner_state, traj_batch = jax.lax.scan(step, runner_state, None, num_steps)

        return traj_batch.obs

    return random_rollout


def update_obs_norm_params(obs_norm_params, obs):

    batch_mean = jnp.mean(obs, axis=0)
    batch_var = jnp.var(obs, axis=0)
    batch_count = obs.shape[0]

    delta = batch_mean - obs_norm_params.mean
    tot_count = obs_norm_params.count + batch_count

    new_mean = obs_norm_params.mean + delta * batch_count / tot_count
    m_a = obs_norm_params.var * obs_norm_params.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * obs_norm_params.count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return ObsNormParams(new_count, new_mean, new_var)


def update_rnd_int_norm_params(batch_count, batch_mean, batch_var, rewems, rnd_int_norm_params):

    delta = batch_mean - rnd_int_norm_params.mean
    tot_count = rnd_int_norm_params.count + batch_count

    new_mean = rnd_int_norm_params.mean + delta * batch_count / tot_count
    m_a = rnd_int_norm_params.var * rnd_int_norm_params.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * rnd_int_norm_params.count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return RNDNormIntReturnParams(new_count, new_mean, new_var, rewems)


def byol_minigrid_ppo_update_networks(
    train_state: TrainState,
    pred_state: TrainState,
    target_state: TrainState,
    transitions: BYOLMiniGridTransition,
    init_hstate: jax.Array,
    init_close_hstate: jnp.ndarray,
    init_open_hstate: jnp.ndarray,
    advantages: jax.Array,
    targets: jax.Array,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    update_target_counter: int,
    ema_param: float,
):
    # NORMALIZE ADVANTAGES
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def pred_loss(pred_params, target_params):
        tar_obs = target_state.apply_fn(target_params, transitions.next_obs)
        pred_input = (transitions.prev_bt.squeeze(2), transitions.obs, transitions.prev_action)
        pred_obs, _, _, _ = pred_state.apply_fn(
            pred_params, init_close_hstate, init_open_hstate, pred_input
        )
        pred_norm = (pred_obs) / (jnp.linalg.norm(pred_obs, axis=-1, keepdims=True))
        tar_norm = jax.lax.stop_gradient(
            (tar_obs) / (jnp.linalg.norm(tar_obs, axis=-1, keepdims=True))
        )
        loss = jnp.square(jnp.linalg.norm((pred_norm - tar_norm), axis=-1)) * (1 - transitions.done)
        return loss.mean()

    def _loss_fn(params):
        # RERUN NETWORK
        dist, value, _ = train_state.apply_fn(
            params,
            {
                # [batch_size, seq_len, ...]
                "observation": transitions.obs,
                "prev_action": transitions.prev_action,
                "prev_reward": transitions.prev_reward,
            },
            init_hstate,
        )
        log_prob = dist.log_prob(transitions.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = transitions.value + (value - transitions.value).clip(
            -clip_eps, clip_eps
        )
        value_loss = jnp.square(value - targets)
        value_loss_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()
        # TODO: ablate this!
        # value_loss = jnp.square(value - targets).mean()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - transitions.log_prob)
        actor_loss1 = advantages * ratio
        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
        entropy = dist.entropy().mean()

        total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy
        return total_loss, (value_loss, actor_loss, entropy)

    (loss, (vloss, aloss, entropy)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
        train_state.params
    )
    pred_losses, pred_grads = jax.value_and_grad(pred_loss)(
        pred_state.params,
        target_state.params,
    )
    (loss, vloss, aloss, entropy, pred_losses, grads, pred_grads) = jax.lax.pmean(
        (loss, vloss, aloss, entropy, pred_losses, grads, pred_grads),
        axis_name="devices",
    )

    def update_target(target_state, pred_state, update_target_counter=update_target_counter):
        def true_fun(_):
            # Perform the EMA update
            return update_target_state_with_ema(
                predictor_state=pred_state,
                target_state=target_state,
                ema_param=ema_param,
            )

        def false_fun(_):
            # Return the old target_params unchanged
            return target_state

        # Conditionally update every 10 steps
        return jax.lax.cond(
            update_target_counter % (10 * 16 * 1) == 0,
            true_fun,
            false_fun,
            None,  # The argument passed to true_fun and false_fun, `_` in this case is unused
        )

    update_target_counter += 1
    train_state = train_state.apply_gradients(grads=grads)
    pred_state = pred_state.apply_gradients(grads=pred_grads)
    target_state = update_target(target_state, pred_state, update_target_counter)

    update_info = {
        "total_loss": loss,
        "value_loss": vloss,
        "actor_loss": aloss,
        "entropy": entropy,
        "pred_loss": pred_losses,
    }
    return (train_state, pred_state, target_state, update_target_counter), update_info


def minigrid_ppo_update_networks(
    train_state: TrainState,
    transitions: MiniGridTransition,
    init_hstate: jax.Array,
    advantages: jax.Array,
    targets: jax.Array,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
):
    # NORMALIZE ADVANTAGES
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _loss_fn(params):
        # RERUN NETWORK
        dist, value, _ = train_state.apply_fn(
            params,
            {
                # [batch_size, seq_len, ...]
                "observation": transitions.obs,
                "prev_action": transitions.prev_action,
                "prev_reward": transitions.prev_reward,
            },
            init_hstate,
        )
        log_prob = dist.log_prob(transitions.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = transitions.value + (value - transitions.value).clip(
            -clip_eps, clip_eps
        )
        value_loss = jnp.square(value - targets)
        value_loss_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()
        # TODO: ablate this!
        # value_loss = jnp.square(value - targets).mean()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - transitions.log_prob)
        actor_loss1 = advantages * ratio
        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
        entropy = dist.entropy().mean()

        total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy
        return total_loss, (value_loss, actor_loss, entropy)

    (loss, (vloss, aloss, entropy)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
        train_state.params
    )
    (loss, vloss, aloss, entropy, grads) = jax.lax.pmean(
        (loss, vloss, aloss, entropy, grads), axis_name="devices"
    )
    train_state = train_state.apply_gradients(grads=grads)
    update_info = {
        "total_loss": loss,
        "value_loss": vloss,
        "actor_loss": aloss,
        "entropy": entropy,
    }
    return train_state, update_info


def rnd_minigrid_ppo_update_networks(
    train_state: TrainState,
    pred_state: TrainState,
    target_params: Any,
    _mask_rng: jnp.ndarray,
    transitions: RNDMiniGridTransition,
    rnd_obs: jnp.ndarray,
    init_hstate: jax.Array,
    advantages: jax.Array,
    targets: jax.Array,
    update_prop: float,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
):
    # NORMALIZE ADVANTAGES
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    target = TargetNetwork(256)

    def _rnd_loss(pred_params, rnd_obs):
        tar_feat = target.apply(target_params, rnd_obs.reshape(-1, rnd_obs.shape[-1]))
        pred_feat = pred_state.apply_fn(pred_params, rnd_obs.reshape(-1, rnd_obs.shape[-1]))
        loss = jnp.square(jnp.linalg.norm((pred_feat - tar_feat), axis=1)) / 2
        mask = jax.random.uniform(_mask_rng, (loss.shape[0],))
        mask = (mask < update_prop).astype(jnp.float32)
        loss = loss * mask

        return loss.sum() / jnp.max(jnp.array([mask.sum(), 1]))

    def _loss_fn(params):
        # RERUN NETWORK
        dist, value, _ = train_state.apply_fn(
            params,
            {
                # [batch_size, seq_len, ...]
                "observation": transitions.obs,
                "prev_action": transitions.prev_action,
                "prev_reward": transitions.prev_reward,
            },
            init_hstate,
        )
        log_prob = dist.log_prob(transitions.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = transitions.value + (value - transitions.value).clip(
            -clip_eps, clip_eps
        )
        value_loss = jnp.square(value - targets)
        value_loss_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()
        # TODO: ablate this!
        # value_loss = jnp.square(value - targets).mean()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - transitions.log_prob)
        actor_loss1 = advantages * ratio
        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
        entropy = dist.entropy().mean()

        total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy
        return total_loss, (value_loss, actor_loss, entropy)

    (loss, (vloss, aloss, entropy)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
        train_state.params
    )

    rnd_loss, rnd_grads = jax.value_and_grad(_rnd_loss)(pred_state.params, rnd_obs)
    (loss, vloss, aloss, entropy, rnd_loss, grads, rnd_grads) = jax.lax.pmean(
        (loss, vloss, aloss, entropy, rnd_loss, grads, rnd_grads), axis_name="devices"
    )
    train_state = train_state.apply_gradients(grads=grads)
    pred_state = pred_state.apply_gradients(grads=rnd_grads)
    update_info = {
        "total_loss": loss,
        "value_loss": vloss,
        "actor_loss": aloss,
        "entropy": entropy,
        "rnd_loss": rnd_loss,
    }
    return (
        train_state,
        pred_state,
    ), update_info


class RolloutStats(struct.PyTreeNode):
    reward: jax.Array = jnp.asarray(0.0)
    length: jax.Array = jnp.asarray(0)
    episodes: jax.Array = jnp.asarray(0)


def rnn_rollout(
    rng: jax.Array,
    env,
    env_params,
    train_state: TrainState,
    init_hstate: jax.Array,
    num_consecutive_episodes: int = 1,
) -> RolloutStats:
    def _cond_fn(carry):
        rng, stats, obsv, env_state, prev_action, prev_reward, hstate = carry
        return jnp.less(stats.episodes, num_consecutive_episodes)

    def _body_fn(carry):
        rng, stats, obsv, env_state, prev_action, prev_reward, hstate = carry

        rng, _rng = jax.random.split(rng)
        dist, _, hstate = train_state.apply_fn(
            train_state.params,
            {
                "observation": obsv[None, None, ...],
                "prev_action": prev_action[None, None, ...],
                "prev_reward": prev_reward[None, None, ...],
            },
            hstate,
        )
        action = dist.sample(seed=_rng).squeeze()
        obsv, env_state, reward, _, done, info = env.step(_rng, env_state, action, env_params)

        stats = stats.replace(
            reward=stats.reward + reward,
            length=stats.length + 1,
            episodes=stats.episodes + done,
        )
        carry = (rng, stats, obsv, env_state, action, reward, hstate)
        return carry

    obsv, env_state = env.reset(rng, env_params)
    prev_action = jnp.asarray(0)
    prev_reward = jnp.asarray(0)
    init_carry = (rng, RolloutStats(), obsv, env_state, prev_action, prev_reward, init_hstate)

    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    return final_carry[1]


def ppo_rollout(
    rng: jax.Array,
    env,
    env_params,
    train_state: TrainState,
    num_consecutive_episodes: int = 1,
) -> RolloutStats:
    def _cond_fn(carry):
        rng, stats, obsv, env_state = carry
        return jnp.less(stats.episodes, num_consecutive_episodes)

    def _body_fn(carry):
        rng, stats, obsv, env_state = carry

        rng, _rng = jax.random.split(rng)
        dist, _, hstate = train_state.apply_fn(
            train_state.params,
            obsv,
        )
        action = dist.sample(seed=_rng).squeeze()
        obsv, env_state, reward, done, _, info = env.step(_rng, env_state, action, env_params)

        stats = stats.replace(
            reward=stats.reward + reward,
            length=stats.length + 1,
            episodes=stats.episodes + done,
        )
        carry = (rng, stats, obsv, env_state, action, reward)
        return carry

    obsv, env_state = env.reset(rng, env_params)
    init_carry = (rng, RolloutStats(), obsv, env_state)

    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    return final_carry[1]


def lifetime_return(life_rewards, lifetime_gamma, reverse=True):
    life_rewards = life_rewards.reshape(-1, life_rewards.shape[2])

    def returns_cal(returns, reward):
        returns = returns * lifetime_gamma + reward
        return returns, returns

    single_return, _ = jax.lax.scan(
        returns_cal, jnp.zeros(life_rewards.shape[-1]), life_rewards, reverse=reverse
    )
    return single_return


def rnd_normalise_int_rewards(traj_batch, rnd_int_return_norm_params, int_gamma, rew_hist):
    def _multiply_rewems_w_dones(rewems, dones_row):
        rewems = rewems * (1 - dones_row)
        return rewems, rewems

    def _update_rewems(rewems, int_reward_row):
        rewems = rewems * int_gamma + int_reward_row
        return rewems, rewems

    # Shape (num_steps, num_envs)
    int_reward = traj_batch.int_reward

    # Shape (num_envs,num_steps)
    int_reward_transpose = jnp.transpose(int_reward)

    rewems, _ = jax.lax.scan(
        _multiply_rewems_w_dones,
        rnd_int_return_norm_params.rewems,
        jnp.transpose(traj_batch.done),  # Shape (num_envs,num_steps)
    )

    rewems, _ = jax.lax.scan(_update_rewems, rewems, int_reward_transpose)

    batch_count = len(rewems)
    batch_mean = rewems.mean()
    batch_var = jnp.var(rewems)

    rnd_int_return_norm_params = update_rnd_int_norm_params(
        batch_count, batch_mean, batch_var, rewems, rnd_int_return_norm_params
    )
    norm_int_reward = int_reward / jnp.sqrt(rnd_int_return_norm_params.var + 1e-8)
    rew_hist /= jnp.sqrt(rnd_int_return_norm_params.var + 1e-8)
    return norm_int_reward, rnd_int_return_norm_params, rew_hist


def rnd_normalise_ext_rewards(traj_batch, rnd_ext_return_norm_params, ext_gamma, rew_hist):
    def _multiply_rewems_w_dones(rewems, dones_row):
        rewems = rewems * (1 - dones_row)
        return rewems, rewems

    def _update_rewems(rewems, ext_reward_row):
        rewems = rewems * ext_gamma + ext_reward_row
        return rewems, rewems

    # Shape (num_steps, num_envs)
    ext_reward = traj_batch.norm_reward

    # Shape (num_envs,num_steps)
    ext_reward_transpose = jnp.transpose(ext_reward)

    rewems, _ = jax.lax.scan(
        _multiply_rewems_w_dones,
        rnd_ext_return_norm_params.rewems,
        jnp.transpose(traj_batch.done),  # Shape (num_envs,num_steps)
    )

    rewems, _ = jax.lax.scan(_update_rewems, rewems, ext_reward_transpose)

    batch_count = len(rewems)
    batch_mean = rewems.mean()
    batch_var = jnp.var(rewems)

    rnd_ext_return_norm_params = update_rnd_int_norm_params(
        batch_count, batch_mean, batch_var, rewems, rnd_ext_return_norm_params
    )
    norm_ext_reward = ext_reward / jnp.sqrt(rnd_ext_return_norm_params.var + 1e-8)
    rew_hist /= jnp.sqrt(rnd_ext_return_norm_params.var + 1e-8)
    return norm_ext_reward, rnd_ext_return_norm_params, rew_hist


def process_output_general(output):
    """
    Process the output dictionary from the training function.

    For every key that contains 'loss' or 'state', the function will unreplicate the value using Flax's unreplicate function.
    For all other keys, it will compute the mean along axis 0. If the value is a nested dictionary, it will recursively process it.

    Args:
        output (dict): The dictionary returned by the training function, containing various metrics.

    Returns:
        dict: A processed dictionary with unreplicated or averaged values as described.
    """

    def process_value(value):
        if isinstance(value, dict):
            return {k: process_value(v) for k, v in value.items()}
        else:
            return jnp.mean(value, axis=0)

    processed_output = {}
    for key, value in output.items():
        if "loss" in key or "state" in key:
            processed_output[key] = unreplicate(value)
        else:
            processed_output[key] = process_value(value)
    return processed_output


def byol_update_reward_prior_norm(norm_int_reward, byol_reward_norm_params, rew_norm_param):
    new_mu_l = (
        rew_norm_param * byol_reward_norm_params.mu_l
        + (1 - rew_norm_param) * norm_int_reward.mean()
    )
    return BYOLRewardNorm(
        byol_reward_norm_params.ema_mean,
        byol_reward_norm_params.ema_mean_sq,
        byol_reward_norm_params.c,
        new_mu_l,
    )


def byol_update_reward_norm_params(byol_reward_norm_params, int_reward, rew_norm_param):
    r_c = int_reward.mean()
    r_c_sq = jnp.square(int_reward).mean()
    new_ema_mean = rew_norm_param * byol_reward_norm_params.ema_mean + (1 - rew_norm_param) * r_c
    new_ema_mean_sq = (
        rew_norm_param * byol_reward_norm_params.ema_mean_sq + (1 - rew_norm_param) * r_c_sq
    )
    new_c = byol_reward_norm_params.c + 1

    return BYOLRewardNorm(new_ema_mean, new_ema_mean_sq, new_c, byol_reward_norm_params.mu_l)


def byol_normalize_prior_int_rewards(
    int_reward, byol_reward_norm_params, rew_norm_param, rew_hist, prior: bool = True
):
    # Update reward normalization parameters
    byol_reward_norm_params = byol_update_reward_norm_params(
        byol_reward_norm_params, int_reward, rew_norm_param
    )
    # Compute the adjusted EMA mean and mean square
    mu_r = byol_reward_norm_params.ema_mean / (1 - (rew_norm_param**byol_reward_norm_params.c))
    mu_r_sq = byol_reward_norm_params.ema_mean_sq / (
        1 - (rew_norm_param**byol_reward_norm_params.c)
    )
    # Compute standard deviation
    mu_array = jnp.array([0, mu_r_sq - jnp.square(mu_r)])
    sigma_r = jnp.sqrt(jnp.max(mu_array) + 1e-8)  # 1e-8 as a small numerical regularization
    # Normalize intrinsic reward
    norm_int_reward = int_reward / sigma_r
    rew_hist /= sigma_r

    # Update prior normalization
    byol_reward_norm_params = byol_update_reward_prior_norm(
        norm_int_reward, byol_reward_norm_params, rew_norm_param
    )

    def prior_norm_step(norm_int_reward, byol_reward_norm_params, rew_hist):
        # Compute prior normalized intrinsic reward
        prior_norm_int_reward = jnp.maximum(norm_int_reward - byol_reward_norm_params.mu_l, 0)
        rew_hist = jnp.maximum(rew_hist - byol_reward_norm_params.mu_l, 0)
        return prior_norm_int_reward, byol_reward_norm_params, rew_hist

    def no_prior_norm_step(norm_int_reward, byol_reward_norm_params, rew_hist):
        return norm_int_reward, byol_reward_norm_params, rew_hist

    prior_norm_int_reward, byol_reward_norm_params, rew_hist = jax.lax.cond(
        prior,
        prior_norm_step,
        no_prior_norm_step,
        norm_int_reward,
        byol_reward_norm_params,
        rew_hist,
    )

    return prior_norm_int_reward, byol_reward_norm_params, rew_hist


def update_target_state_with_ema(predictor_state, target_state, ema_param):
    """Update the target network parameters with EMA for encoder layers only."""

    def update_encoder_params(target_params, predictor_params, param_names):
        updated_params = {}
        for name, target_param in target_params.items():
            if name in param_names:
                updated_params[name] = (
                    ema_param * target_param + (1 - ema_param) * predictor_params[name]
                )
            else:
                updated_params[name] = target_param
        return updated_params

    # Names of the encoder layers in the predictor network
    encoder_layer_names = ["encoder_layer_1", "encoder_layer_2"]

    # Get the new encoder parameters using EMA
    new_encoder_params = update_encoder_params(
        target_state.params, predictor_state.params, encoder_layer_names
    )

    # Replace the target state's parameters with the new encoder parameters
    new_target_params = target_state.params.copy()
    new_target_params.update(new_encoder_params)
    target_state = target_state.replace(params=new_target_params)

    return target_state


def compress_output_for_reasoning(output, minigrid=False):
    output_compressed = {}

    for key, value in output.items():
        if key == "train_states":
            # Keep train_states as is
            output_compressed[key] = value
        elif "loss" in key.lower():
            if minigrid:
                # Leave loss as is for minigrid
                output_compressed[key] = value
            else:
                # Apply double mean for loss values in non-minigrid case
                output_compressed[key] = value.mean(-1).mean(-1)
        elif key == "metrics":
            # Apply mean to each value in the metrics dictionary
            output_compressed[key] = {k: v.mean(-1) for k, v in value.items()}
        else:
            # For all other keys, take mean along the last axis
            output_compressed[key] = value.mean(-1)

    return output_compressed


def create_adjacent_pairs(candidate_params):
    def split_adjacent(arr):
        # Ensure the first dimension is even
        assert arr.shape[0] % 2 == 0, "First dimension must be even"
        return arr.reshape(-1, 2, *arr.shape[1:])

    # Split each parameter array into pairs
    paired_params = jax.tree.map(split_adjacent, candidate_params)

    # Get the number of pairs
    num_pairs = jax.tree.leaves(paired_params)[0].shape[0]

    # Function to extract a single pair from the paired structure
    def extract_pair(paired_tree, idx):
        return jax.tree.map(lambda x: x[idx], paired_tree)

    # Create a list of parameter pairs
    param_pairs = [extract_pair(paired_params, i) for i in range(num_pairs)]

    return param_pairs


def reshape_adjacent_pairs(candidate_params):
    def combine_adjacent(arr):
        # Ensure the first dimension is even
        assert arr.shape[0] % 2 == 0, "First dimension must be even"
        new_shape = (arr.shape[0] // 2, 2, *arr.shape[1:])
        return arr.reshape(new_shape)

    # Reshape each parameter array to combine adjacent pairs
    reshaped_params = jax.tree_util.tree_map(combine_adjacent, candidate_params)

    return reshaped_params


def reorder_antithetic_pairs(params, population_size):
    """
    Reorder parameters to make antithetic samples adjacent in the population.

    Parameters:
    - params: A pytree containing population parameters (e.g., dict, list, or array).
    - population_size: The total number of individuals in the population (must be even).

    Returns:
    - Reordered pytree where antithetic pairs are adjacent.
    """
    # Ensure population_size is even for antithetic pairing
    assert population_size % 2 == 0, "population_size must be even for antithetic pairing."

    # Generate indices to reorder the population
    idxs = jnp.concatenate(
        [jnp.array([i, i + population_size // 2]) for i in range(population_size // 2)]
    )

    # Reorder the parameters based on the calculated indices
    reordered_params = jax.tree_util.tree_map(lambda arr: arr[idxs], params)

    return reordered_params
