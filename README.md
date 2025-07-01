# Meta-learning Curiosity Algorithms

## Overview

This repository aims to develop **intrinsic motivational systems that can generalise across different environments without needing to be redesigned when the agent changes environments**. The ultimate goal is to create curiosity-driven agents that can adapt their exploration strategies across diverse domains.

### Current Progress: Adaptive Intrinsic Reward Weighting

As a first step toward this vision, this repository implements a meta-learning approach for automatically tuning intrinsic reward weightings (λ) in curiosity-driven reinforcement learning within grid-world environments.

#### Key Features
- **Reward Combiner**: A recurrent neural network that learns to dynamically weight intrinsic rewards from curiosity algorithms
- **Multi-task Training**: Uses evolutionary strategies (ES) to train across multiple XLand-MiniGrid environments
- **Curiosity Algorithms**: Implements Random Network Distillation (RND) and BYOL-Explore

The current approach demonstrates potential for generalisation across different grid sizes and task objectives within navigation and door-key interaction environments.

## Built Upon
- **[PureJAXRL](https://github.com/luchris429/purejaxrl)** - JAX-based RL implementations
- **[XLand-MiniGrid](https://github.com/dunnolab/xland-minigrid)** - Scalable meta-RL environments
- **[Groove](https://github.com/EmptyJackson/groove)** - Discovering temporally-aware RL algorithms (adapted multi-task ES approach)

## Installation
```bash
pip install -r requirements.txt
