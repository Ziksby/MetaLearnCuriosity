# Taken from:
# https://github.com/corl-team/xland-minigrid/blob/main/training/train_single_task.py

import gc
import os
import shutil
import time

import jax
import jax.numpy as jnp
import jax.tree_util
from evosax import OpenES
from flax.jax_utils import replicate
from tqdm import tqdm

import wandb
from MetaLearnCuriosity.agents.nn import RewardCombiner
from MetaLearnCuriosity.checkpoints import Save
from MetaLearnCuriosity.compile_rnd_brax_trains import compile_rnd_fns
from MetaLearnCuriosity.logger import WBLogger
from MetaLearnCuriosity.utils import (
    create_adjacent_pairs,
    process_output_general,
    reorder_antithetic_pairs,
)

environments = [
    # "ant",
    # "halfcheetah",
    # "hopper",
    # "humanoid",
    # "humanoidstandup",
    # "inverted_pendulum",
    # "inverted_double_pendulum",
    # "pusher",
    # "reacher",
    # "walker2d",
]

config = {
    "RUN_NAME": "DELAY_RC_CNN_brax",
    "SEED": 89,
    "NUM_SEEDS": 1,
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
    "INT_GAMMA": 0.99,
    "PRED_LR": 1e-3,
    # "INT_LAMBDA": 0.02,
    "HIST_LEN": 128,
    "POP_SIZE": 64,
    "RC_SEED": 23,
    "ES_SEED": 98647,
    "NUM_GENERATIONS": 48,
}
step_intervals = [3, 10, 20, 30]


reward_combiner_network = RewardCombiner()

rc_params_pholder = reward_combiner_network.init(
    jax.random.PRNGKey(config["RC_SEED"]), jnp.zeros((1, 128, 2))
)
es_rng = jax.random.PRNGKey(config["ES_SEED"])
strategy = OpenES(
    popsize=config["POP_SIZE"],
    pholder_params=rc_params_pholder,
    opt_name="adam",
    lrate_decay=1,
    sigma_decay=0.999,
    sigma_init=0.04,
    n_devices=1,
    maximize=True,
)
group = "reward_combiners"
tags = ["meta-learner", "fitness"]
name = f'{config["RUN_NAME"]}'
es_rng, es_rng_init = jax.random.split(es_rng)
es_params = strategy.default_params
es_state = strategy.initialize(es_rng_init, es_params)
train_fns, make_seeds = compile_rnd_fns(config=config)
rng = jax.random.PRNGKey(config["SEED"])
fit_log = wandb.init(
    project="MetaLearnCuriosity",
    config=config,
    group=group,
    tags=tags,
    name=f"{name}_fitness",
)
env_name = "hopper"
for gen in tqdm(range(config["NUM_GENERATIONS"]), desc="Processing Generations"):
    begin_gen = time.time()
    es_rng, es_rng_ask = jax.random.split(es_rng)
    x, es_state = strategy.ask(es_rng_ask, es_state, es_params)
    x = reorder_antithetic_pairs(x, config["POP_SIZE"])
    pairs = create_adjacent_pairs(x)
    fitness = []
    raw_fitness_dict = {step_int: [] for step_int in step_intervals}
    int_lambda_dict = {step_int: [] for step_int in step_intervals}

    for pair in pairs:
        t = time.time()
        rng, env_key = jax.random.split(rng)
        rng, rng_seeds = jax.random.split(rng)
        rng_train = jax.random.split(rng_seeds, config["NUM_SEEDS"])
        index = jax.random.choice(env_key, len(step_intervals))
        step_int = step_intervals[index]

        # config, env, env_params = make_config_env(config, env_name)
        print(f"Training in {env_name}_{step_int} in gen ", gen)

        # setting up the RL agents.
        (
            rng_train,
            train_state,
            pred_state,
            target_params,
            init_rnd_obs,
            ext_reward_hist,
            int_reward_hist,
        ) = make_seeds[step_int](rng_train)

        # duplicating here for pmap
        train_state = replicate(train_state, jax.local_devices())
        pred_state = replicate(pred_state, jax.local_devices())
        target_params = replicate(target_params, jax.local_devices())
        pair = replicate(pair, jax.local_devices())
        init_rnd_obs = replicate(init_rnd_obs, jax.local_devices())

        ext_reward_hist = replicate(ext_reward_hist, jax.local_devices())
        int_reward_hist = replicate(int_reward_hist, jax.local_devices())
        t = time.time()

        output = jax.block_until_ready(
            train_fns[step_int](
                rng_train,
                pair,
                train_state,
                pred_state,
                target_params,
                init_rnd_obs,
                ext_reward_hist,
                int_reward_hist,
            )
        )
        output = process_output_general(output)
        raw_episode_return = output["rewards"].mean(-1)  # This is the raw fitness
        int_lambdas = output["int_lambdas"].mean(
            -1
        )  # Get the int_lambdas and average across episodes
        episode_returns = output["episode_returns"].mean(-1)
        raw_fitness_dict[step_int].append(raw_episode_return)  # Store raw fitness
        int_lambda_dict[step_int].append(int_lambdas)  # Store int_lambdas
        print("Here is the episode return of the pair:", episode_returns)
        print("Here is the int_lambda of the pair:", int_lambdas)
        print("Here is the fitness of the pair:", raw_episode_return)
        binary_fitness = jnp.where(raw_episode_return == jnp.max(raw_episode_return), 1.0, 0.0)
        fitness.append(binary_fitness)
        print(f"Time for the Pair in {env_name}_{step_int} is {(time.time()-t)/60}")

    fitness = jnp.array(fitness).flatten()
    es_state = strategy.tell(x, fitness, es_state, es_params)

    # Save the state
    checkpoint_directory = f'MLC_logs/flax_ckpt/Reward_Combiners/Multi_task/{config["RUN_NAME"]}'
    path = os.path.abspath(checkpoint_directory)
    details = (es_state, config, es_rng, rng)
    Save(path, details)
    print("Generation ", gen, "Time:", (time.time() - begin_gen) / 60)

    # logging now to W&Bs
    for step_int in step_intervals:
        raw_fitness = raw_fitness_dict[step_int]
        int_lambdas = int_lambda_dict[step_int]

        if len(raw_fitness) > 0:
            # Convert to numpy arrays for easier manipulation
            raw_fitness_array = jnp.array(raw_fitness)
            int_lambda_array = jnp.array(int_lambdas)

            # Find the best individual based on raw fitness
            best_idx = jnp.argmax(raw_fitness_array)

            fit_log.log(
                {
                    f"{env_name}_{step_int}_mean_fitness": raw_fitness_array.mean(),
                    f"{env_name}_{step_int}_best_fitness": jnp.max(raw_fitness_array),
                    f"{env_name}_{step_int}_mean_lambda": int_lambda_array.mean(),  # Average lambda across generation
                    f"{env_name}_{step_int}_best_lambda": int_lambda_array[best_idx][
                        0
                    ],  # Lambda of best individual
                }
            )
        else:
            print(f"Warning: No fitness data for Breakout_{step_int} in generation {gen}")
            fit_log.log(
                {
                    f"{env_name}_{step_int}_mean_fitness": 0.0,
                    f"{env_name}_{step_int}_best_fitness": 0.0,
                    f"{env_name}_{step_int}_mean_lambda": 0.0,
                    f"{env_name}_{step_int}_best_lambda": 0.0,
                }
            )
    gc.collect()
fit_log.finish()

logger = WBLogger(
    config=config,
    group="meta_learners",
    tags=["multi_task", "reward-combiner"],
    name=config["RUN_NAME"],
)
logger.save_artifact(path)
shutil.rmtree(path)
