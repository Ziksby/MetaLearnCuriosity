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
python MetaLearnCuriosity/agents/curious_agents/single_value_head_agents/rnd_minigrid.py

# Check if the previous command was successful
if [ $? -eq 0 ]; then
    echo "rnd_minigrid.py ran successfully."
else
    echo "rnd_minigrid.py failed to run."
    exit 1
fi

# Add more scripts as needed
python MetaLearnCuriosity/agents/curious_agents/single_value_head_agents/byol_minigrid.py
# Check if the previous command was successful
if [ $? -eq 0 ]; then
    echo "byol_minigrid.py ran successfully."
else
    echo "byol_minigrid.py failed to run."
    exit 1
fi
# Print a message when all scripts have run
echo "All scripts have run."
