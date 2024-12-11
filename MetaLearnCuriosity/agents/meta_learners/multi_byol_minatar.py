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
from MetaLearnCuriosity.agents.nn import RCRNN, RewardCombiner
from MetaLearnCuriosity.checkpoints import Restore, Save
from MetaLearnCuriosity.compile_gymnax_trains import compile_fns
from MetaLearnCuriosity.logger import WBLogger
from MetaLearnCuriosity.utils import (
    create_adjacent_pairs,
    get_latest_commit_hash,
    process_output_general,
    reorder_antithetic_pairs,
)

config = {
    "RUN_NAME": "rc_cnn_minatar_default_delayed_breakout_EPISODE_RETURNS",
    "SEED": 42 * 556,
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
    "POP_SIZE": 64,
    "ES_SEED": 7 * 644,
    "HIST_LEN": 128,
    "RC_SEED": 23 * 89,
    "NUM_GENERATIONS": 48,
    # "INT_LAMBDA": 0.001,
    "ENV_KEY": 102,
}
commit_hash = get_latest_commit_hash()

config["COMMIT_HARSH"] = commit_hash
reward_combiner_network = RewardCombiner()
env_name = "Breakout-MinAtar"
rc_params_pholder = reward_combiner_network.init(
    jax.random.PRNGKey(config["RC_SEED"]), jnp.zeros((1, config["HIST_LEN"], 2))
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

# strategy = OpenES(
#     popsize=config["POP_SIZE"],
#     pholder_params=rc_params_pholder,
#     opt_name="adam",
#     lrate_init=1e-2,
#     lrate_decay=0.999,
#     lrate_limit=1e-5,
#     sigma_init=0.1,
#     sigma_decay=1.0,
#     sigma_limit=0.1,
#     n_devices=1,
#     maximize=True,
# )
group = "reward_combiners"
tags = ["meta-learner", "fitness"]
name = f'{config["RUN_NAME"]}'
es_rng, es_rng_init = jax.random.split(es_rng)
es_params = strategy.default_params
es_state = strategy.initialize(es_rng_init, es_params)

es_stuff = Restore(
    "/home/batsy/MetaLearnCuriosity/MLC_logs/flax_ckpt/Reward_Combiners/Multi_task/rc_rnn_minatar_default_delayed_breakout_SAVED"
)
es_state_saved, _ = es_stuff
print(es_state_saved)
print()
print(es_state)
print()
opt_state = es_state.opt_state.replace(
    lrate=es_state_saved["opt_state"]["lrate"],
    m=es_state_saved["opt_state"]["m"],
    v=es_state_saved["opt_state"]["v"],
    n=es_state_saved["opt_state"]["n"],
    last_grads=es_state_saved["opt_state"]["last_grads"],
    gen_counter=es_state_saved["opt_state"]["gen_counter"],
)
es_state = es_state.replace(
    mean=es_state_saved["mean"],
    sigma=es_state_saved["sigma"],
    opt_state=opt_state,
    best_member=es_state_saved["best_member"],
    best_fitness=es_state_saved["best_fitness"],
    gen_counter=es_state_saved["gen_counter"],
)
print("Now matched,", es_state, "\n")

train_fns, make_seeds = compile_fns(config=config)
rng = jax.random.PRNGKey(config["SEED"])
fit_log = wandb.init(
    project="MetaLearnCuriosity",
    config=config,
    group=group,
    tags=tags,
    name=f"{name}",
)
# step_intervals = [0.1,0.3,0.5,0.8]
step_intervals = [3, 10, 20, 30]

for gen in tqdm(
    range(config["NUM_GENERATIONS"] - es_state_saved["gen_counter"]), desc="Processing Generations"
):
    begin_gen = time.time()
    es_rng, es_rng_ask = jax.random.split(es_rng)
    x, es_state = strategy.ask(es_rng_ask, es_state, es_params)
    x = reorder_antithetic_pairs(x, config["POP_SIZE"])
    pairs = create_adjacent_pairs(x)
    fitness = []
    raw_fitness_dict = {step_int: [] for step_int in step_intervals}
    int_lambda_dict = {
        step_int: [] for step_int in step_intervals
    }  # New dictionary for int_lambdas

    for pair in pairs:
        t = time.time()
        rng, env_key = jax.random.split(rng)
        rng, rng_seeds = jax.random.split(rng)
        rng_train = jax.random.split(rng_seeds, config["NUM_SEEDS"])
        index = jax.random.choice(env_key, len(step_intervals))
        step_int = step_intervals[index]

        print(f"Training in {env_name}_{step_int} in gen ", gen)

        (
            rng_train,
            train_state,
            pred_state,
            target_state,
            init_bt,
            close_init_hstate,
            open_init_hstate,
            init_action,
            ext_reward_hist,
            int_reward_hist,
        ) = make_seeds[step_int](rng_train)

        # duplicating here for pmap
        open_init_hstate = replicate(open_init_hstate, jax.local_devices())
        close_init_hstate = replicate(close_init_hstate, jax.local_devices())
        train_state = replicate(train_state, jax.local_devices())
        pred_state = replicate(pred_state, jax.local_devices())
        target_state = replicate(target_state, jax.local_devices())
        init_bt = replicate(init_bt, jax.local_devices())
        init_action = replicate(init_action, jax.local_devices())
        pair = replicate(pair, jax.local_devices())
        ext_reward_hist = replicate(ext_reward_hist, jax.local_devices())
        int_reward_hist = replicate(int_reward_hist, jax.local_devices())
        t = time.time()

        output = jax.block_until_ready(
            train_fns[step_int](
                rng_train,
                pair,
                train_state,
                pred_state,
                target_state,
                init_bt,
                close_init_hstate,
                open_init_hstate,
                init_action,
                ext_reward_hist,
                int_reward_hist,
            )
        )
        output = process_output_general(output)
        raw_episode_return = output["episode_returns"].mean(-1)  # This is the raw fitness
        int_lambdas = output["int_lambdas"].mean(
            -1
        )  # Get the int_lambdas and average across episodes
        episode_returns = output["episode_returns"].mean(-1)
        raw_fitness_dict[step_int].append(raw_episode_return)  # Store raw fitness
        int_lambda_dict[step_int].append(int_lambdas)  # Store int_lambdas

        binary_fitness = jnp.where(raw_episode_return == jnp.max(raw_episode_return), 1.0, 0.0)
        fitness.append(binary_fitness)
        print("Here is the episode return of the pair:", episode_returns)
        print("Here is the int_lambda of the pair:", int_lambdas)
        print("Here is the fitness of the pair:", raw_episode_return)
        print(f"Time for the Pair in {env_name}_{step_int} is {(time.time()-t)/60}")

    fitness = jnp.array(fitness).flatten()
    es_state = strategy.tell(x, fitness, es_state, es_params)

    # Save the state
    checkpoint_directory = f'MLC_logs/flax_ckpt/Reward_Combiners/Multi_task/{config["RUN_NAME"]}'
    path = os.path.abspath(checkpoint_directory)
    details = (es_state, config)
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
                    f"Breakout_{step_int}_mean_fitness_episode": raw_fitness_array.mean(),
                    f"Breakout_{step_int}_best_fitness_episode": jnp.max(raw_fitness_array),
                    f"Breakout_{step_int}_mean_lambda": int_lambda_array.mean(),  # Average lambda across generation
                    f"Breakout_{step_int}_best_lambda": int_lambda_array[best_idx][
                        0
                    ],  # Lambda of best individual
                }
            )
        else:
            print(f"Warning: No fitness data for Breakout_{step_int} in generation {gen}")
            fit_log.log(
                {
                    f"Breakout_{step_int}_mean_fitness_episode": 0.0,
                    f"Breakout_{step_int}_best_fitness_episode": 0.0,
                    f"Breakout_{step_int}_mean_lambda": 0.0,
                    f"Breakout_{step_int}_best_lambda": 0.0,
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
