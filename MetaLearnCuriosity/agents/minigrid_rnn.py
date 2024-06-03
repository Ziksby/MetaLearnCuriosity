# Taken from:
# https://github.com/corl-team/xland-minigrid/blob/main/training/train_single_task.py

import os

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState

import wandb
from MetaLearnCuriosity.agents.nn import MiniGridActorCriticRNN
from MetaLearnCuriosity.checkpoints import Save
from MetaLearnCuriosity.logger import WBLogger
from MetaLearnCuriosity.utils import MiniGridTransition as Transition
from MetaLearnCuriosity.utils import (
    calculate_gae,
    minigrid_ppo_update_networks,
    rnn_rollout,
)
from MetaLearnCuriosity.wrappers import (
    FlattenObservationWrapper,
    LogWrapper,
    MiniGridGymnax,
    VecEnv,
)

jax.config.update("jax_threefry_partitionable", True)

config = {
    "NUM_SEEDS": 1,
    "PROJECT": "xminigrid",
    "RUN_NAME": "single-task-ppo-flatten",
    "ENV_NAME": "MiniGrid-Empty-16x16",
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
    "TOTAL_TIMESTEPS": 10_000_000,
    "LR": 0.001,
    "CLIP_EPS": 0.2,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "EVAL_EPISODES": 80,
    "SEED": 42,
}
rng = jax.random.PRNGKey(config["SEED"])
if config["NUM_SEEDS"] > 1:
    rng = jax.random.split(rng, config["NUM_SEEDS"])
num_devices = jax.local_device_count()
config["NUM_ENVS_PER_DEVICE"] = config["NUM_ENVS"] // num_devices
config["TOTAL_TIMESTEPS_PER_DEVICE"] = config["TOTAL_TIMESTEPS"] // num_devices
config["EVAL_EPISODES_PER_DEVICE"] = config["EVAL_EPISODES"] // num_devices
assert config["NUM_ENVS"] % num_devices == 0
env = MiniGridGymnax(config["ENV_NAME"])
env_params = env._env_params
env_eval = MiniGridGymnax(config["ENV_NAME"])
if not config["USE_CNNS"]:
    env = FlattenObservationWrapper(env)
    observations_shape = env.observation_space(env_params).shape
else:
    observations_shape = env.observation_space(env_params).shape[0]
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


def make_train(rng):
    rng, _rng = jax.random.split(rng)

    num_actions = env.action_space(env_params).n

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac
        # INIT network

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

    network_params = network.init(_rng, init_obs, init_hstate)
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.inject_hyperparams(optax.adam)(learning_rate=linear_schedule, eps=1e-8),  # eps=1e-5
    )
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
    # env = VecEnv(env)

    return init_hstate, train_state, rng


def train(rng, init_hstate, train_state):
    rng, _rng = jax.random.split(rng)

    reset_rng = jax.random.split(_rng, config["NUM_ENVS_PER_DEVICE"])

    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
    prev_action = jnp.zeros(config["NUM_ENVS_PER_DEVICE"], dtype=jnp.int32)
    prev_reward = jnp.zeros(config["NUM_ENVS_PER_DEVICE"])

    # TRAIN LOOP

    def _update_step(runner_state, _):

        # COLLECT TRAJECTORIES

        def _env_step(runner_state, _):
            (
                rng,
                train_state,
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
            obsv, env_state, reward, _, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                rng_step, env_state, action, env_params
            )
            transition = Transition(
                done=done,
                action=action,
                value=value,
                reward=reward,
                log_prob=log_prob,
                obs=prev_obs,
                prev_action=prev_action,
                prev_reward=prev_reward,
                info=info,
            )
            runner_state = (rng, train_state, env_state, obsv, action, reward, hstate)
            return runner_state, transition

        initial_hstate = runner_state[-1]
        runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

        # CALCULATE ADVANTAGE

        rng, train_state, env_state, prev_obs, prev_action, prev_reward, hstate = runner_state

        _, last_val, _ = train_state.apply_fn(
            train_state.params,
            {
                "observation": prev_obs[:, None],
                "prev_action": prev_action[:, None],
                "prev_reward": prev_reward[:, None],
            },
            hstate,
        )

        advantages, targets = calculate_gae(
            transitions, last_val.squeeze(1), config["GAMMA"], config["GAE_LAMBDA"]
        )

        # UPDATE NETWORK
        def _update_epoch(update_state, _):
            def _update_minbatch(train_state, batch_info):
                init_hstate, transitions, advantages, targets = batch_info
                new_train_state, update_info = minigrid_ppo_update_networks(
                    train_state=train_state,
                    transitions=transitions,
                    init_hstate=init_hstate.squeeze(1),
                    advantages=advantages,
                    targets=targets,
                    clip_eps=config["CLIP_EPS"],
                    vf_coef=config["VF_COEF"],
                    ent_coef=config["ENT_COEF"],
                )
                return new_train_state, update_info

            rng, train_state, init_hstate, transitions, advantages, targets = update_state

            # MINIBATCHES PREPARATION
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, config["NUM_ENVS_PER_DEVICE"])
            # [seq_len, batch_size, ...]
            batch = (init_hstate, transitions, advantages, targets)
            # [batch_size, seq_len, ...], as our model assumes
            batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

            shuffled_batch = jtu.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
            # [num_minibatches, minibatch_size, ...]
            minibatches = jtu.tree_map(
                lambda x: jnp.reshape(x, (config["NUM_MINIBATCHES"], -1) + x.shape[1:]),
                shuffled_batch,
            )
            train_state, update_info = jax.lax.scan(_update_minbatch, train_state, minibatches)

            update_state = (rng, train_state, init_hstate, transitions, advantages, targets)
            return update_state, update_info

        # [seq_len, batch_size, num_layers, hidden_dim]
        init_hstate = initial_hstate[None, :]
        update_state = (rng, train_state, init_hstate, transitions, advantages, targets)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )

        # averaging over minibatches then over epochs
        loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)

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

        rng, train_state = update_state[:2]
        traj_batch = update_state[3]
        metric = traj_batch.info
        runner_state = (rng, train_state, env_state, prev_obs, prev_action, prev_reward, hstate)
        return runner_state, (metric, loss_info)

    runner_state = (rng, train_state, env_state, obsv, prev_action, prev_reward, init_hstate)
    runner_state, loss_info = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
    metric, loss = loss_info
    return {
        "runner_state": runner_state,
        "metrics": metric,
        "loss_info": loss,
        "rl_total_loss": loss["total_loss"],
        "rl_value_loss": loss["value_loss"],
        "rl_actor_loss": loss["actor_loss"],
        "rl_entrophy_loss": loss["entropy"],
    }


if config["NUM_SEEDS"] > 1:
    init_hstate, train_state, rng = jax.vmap(make_train)(rng)
    init_hstate = replicate(init_hstate, jax.local_devices())
    train_state = replicate(train_state, jax.local_devices())
    train_fn = jax.vmap(train, in_axes=(0, 0, 0))
    train_fn = jax.pmap(train_fn, axis_name="devices")
    vectorized_split = jax.vmap(
        lambda key: jax.random.split(key, num=jax.device_count()), out_axes=1
    )
    rng = vectorized_split(rng)
    output = jax.block_until_ready(train_fn(rng, init_hstate, train_state))
    output = unreplicate(output)


else:
    init_hstate, train_state, rng = make_train(rng)
    train_state = replicate(train_state, jax.local_devices())
    init_hstate = replicate(init_hstate, jax.local_devices())
    rng = jax.random.split(rng, num=jax.device_count())
    train_fn = jax.pmap(train, axis_name="devices")
    output = jax.block_until_ready(train_fn(rng, init_hstate, train_state))
    output = unreplicate(output)

print(output["rl_total_loss"].shape)
logger = WBLogger(
    config=config,
    group=f"rnn_minigrid/{config['ENV_NAME']}",
    tags=["rnn_minigrid", config["ENV_NAME"]],
    name=config["RUN_NAME"],
)
logger.log_episode_return(output, config["NUM_SEEDS"])
logger.log_rl_loss_minigrid(output, config["NUM_SEEDS"])
output["config"] = config
checkpoint_directory = f'MLC_logs/flax_ckpt/{config["ENV_NAME"]}/{config["RUN_NAME"]}'

# Get the absolute path of the directory
path = os.path.abspath(checkpoint_directory)
Save(path, output)
logger.save_artifact(path)
