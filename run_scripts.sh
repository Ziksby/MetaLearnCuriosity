#!/bin/bash

# Run the first Python script
# python MetaLearnCuriosity/agents/curious_agents/single_value_head_agents/rnd_gymnax.py

# Check if the previous command was successful
# if [ $? -eq 0 ]; then
#     echo "rnd_gymnax.py ran successfully."
# else
#     echo "rnd_gymnax.py failed to run."
#     exit 1
# fi

# Run the second Python script
python MetaLearnCuriosity/agents/continuous_action_ppo.py

# Check if the previous command was successful
if [ $? -eq 0 ]; then
    echo "rnd_brax.py ran successfully."
else
    echo "rnd_brax.py failed to run."
    exit 1
fi

# Add more scripts as needed
python MetaLearnCuriosity/hyperparameter_sweep/rnd_notebook_brax.py
# Check if the previous command was successful
if [ $? -eq 0 ]; then
    echo "byol_brax.py ran successfully."
else
    echo "byol_brax.py failed to run."
    exit 1
fi
# Print a message when all scripts have run
echo "All scripts have run."
