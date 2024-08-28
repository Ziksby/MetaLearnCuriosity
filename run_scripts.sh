#!/bin/bash

# Run the first Python script
python MetaLearnCuriosity/agents/curious_agents/single_value_head_agents/reward_combiner_minatar.py

# Check if the previous command was successful
if [ $? -eq 0 ]; then
    echo "reward_combiner_minatar.py ran successfully."
else
    echo "reward_combiner_minatar.py failed to run."
    exit 1
fi

# Run the second Python script
python MetaLearnCuriosity/agents/curious_agents/single_value_head_agents/reward_combiner_minigrid.py

# Check if the previous command was successful
if [ $? -eq 0 ]; then
    echo "reward_combiner_minigrid.py ran successfully."
else
    echo "reward_combiner_minigrid.py failed to run."
    exit 1
fi

# Add more scripts as needed
python MetaLearnCuriosity/agents/curious_agents/single_value_head_agents/reward_combiner_delayed_brax.py
# Check if the previous command was successful
if [ $? -eq 0 ]; then
    echo "reward_combiner_delayed_brax.py ran successfully."
else
    echo "reward_combiner_delayed_brax.py failed to run."
    exit 1
fi
# Print a message when all scripts have run
echo "All scripts have run."
