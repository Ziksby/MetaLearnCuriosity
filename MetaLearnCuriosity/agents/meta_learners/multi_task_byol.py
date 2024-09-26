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

from MetaLearnCuriosity.agents.nn import RewardCombiner
from MetaLearnCuriosity.checkpoints import Save
from MetaLearnCuriosity.compile_gymnax_trains import compile_fns
from MetaLearnCuriosity.logger import WBLogger
from MetaLearnCuriosity.utils import (
    create_adjacent_pairs,
    process_output_general,
    reorder_antithetic_pairs,
)

environments = [
    # "Asterix-MinAtar",
    "Breakout-MinAtar",
    # "Freeway-MinAtar",
    "SpaceInvaders-MinAtar",
]


config = {
    "RUN_NAME": "rc_meta_default",
    "SEED": 42,
    "NUM_SEEDS": 2,
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
    "PRED_LR": 0.001,
    "REW_NORM_PARAMETER": 0.99,
    "EMA_PARAMETER": 0.99,
    "POP_SIZE": 8,
    "ES_SEED": 7,
    "RC_SEED": 23,
    "NUM_GENERATIONS": 2,
    # "INT_LAMBDA": 0.001,
    "ENV_KEY": 102,
    "TRAIN_ENVS": environments,
}


reward_combiner_network = RewardCombiner()
env_key = jax.random.PRNGKey(config["ENV_KEY"])
rc_params_pholder = reward_combiner_network.init(
    jax.random.PRNGKey(config["RC_SEED"]), jnp.zeros((1, 2))
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

es_rng, es_rng_init = jax.random.split(es_rng)
es_params = strategy.default_params
es_state = strategy.initialize(es_rng_init, es_params)
train_fns, make_seeds = compile_fns()
rng = jax.random.PRNGKey(config["SEED"])

for gen in tqdm(range(config["NUM_GENERATIONS"]), desc="Processing Generations"):
    begin_gen = time.time()
    es_rng, es_rng_ask = jax.random.split(es_rng)
    x, es_state = strategy.ask(es_rng_ask, es_state, es_params)
    x = reorder_antithetic_pairs(x, config["POP_SIZE"])
    pairs = create_adjacent_pairs(x)
    fitness = []

    for pair in pairs:
        t = time.time()
        rng, env_key = jax.random.split(rng)
        rng, rng_seeds = jax.random.split(rng)
        rng_train = jax.random.split(rng_seeds, config["NUM_SEEDS"])
        index = jax.random.choice(env_key, len(environments))
        env_name = environments[index]

        # config, env, env_params = make_config_env(config, env_name)
        print(f"Training in {env_name}")

        # setting up the RL agents.
        (
            rng_train,
            train_state,
            pred_state,
            target_state,
            init_bt,
            close_init_hstate,
            open_init_hstate,
            init_action,
        ) = make_seeds[env_name](rng_train)

        # duplicating here for pmap
        open_init_hstate = replicate(open_init_hstate, jax.local_devices())
        close_init_hstate = replicate(close_init_hstate, jax.local_devices())
        train_state = replicate(train_state, jax.local_devices())
        pred_state = replicate(pred_state, jax.local_devices())
        target_state = replicate(target_state, jax.local_devices())
        init_bt = replicate(init_bt, jax.local_devices())
        init_action = replicate(init_action, jax.local_devices())
        pair = replicate(pair, jax.local_devices())
        t = time.time()

        output = jax.block_until_ready(
            train_fns[env_name](
                rng_train,
                pair,
                train_state,
                pred_state,
                target_state,
                init_bt,
                close_init_hstate,
                open_init_hstate,
                init_action,
            )
        )
        output = process_output_general(output)
        binary_fitness = output["rewards"].mean(-1)
        binary_fitness = jnp.where(binary_fitness == jnp.max(binary_fitness), 1.0, 0.0)
        fitness.append(binary_fitness)
        print(f"Time for the Pair in {env_name} is {(time.time()-t)/60}")

    fitness = jnp.array(fitness).flatten()
    es_state = strategy.tell(x, fitness, es_state, es_params)

    # Save the state
    checkpoint_directory = f'MLC_logs/flax_ckpt/Reward_Combiners/Multi_task/{config["RUN_NAME"]}'
    path = os.path.abspath(checkpoint_directory)
    details = (es_state, config)
    Save(path, details)
    print("Generation ", gen, (time.time() - begin_gen) / 60)
    gc.collect()

logger = WBLogger(
    config=config,
    group="meta_learners",
    tags=["multi_task", "reward-combiner"],
    name=config["RUN_NAME"],
)
logger.save_artifact(path)
shutil.rmtree(path)
