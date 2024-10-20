# Independent Node Exploration and Proximal Trajectory

This repository is an implementation of proximal policy optimization (PPO) in a multi-agent environment, specifically targeting multimodal alignment of single-cell data.  The methodology is based off of OpenAI's multi-agent reinforcement learning (RL) [paper](https://arxiv.org/abs/1909.07528) implementing hide-and-seek, adapted to biological and similar information.  The codebase itself is quite flexible, and can be extended to environments in other domains.

## Examples

![INEPT trained on 50 cells of MMD-MA simulation data, generalized to the whole dataset (300 cells) and run for 200 steps](./plots/MMD-MA_integration.gif)

![Same simulation with single perturbed cell traced from steady-state position](./plots/MMD-MA_trajectory.gif)
