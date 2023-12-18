# Pacman AI

Trained a Pacman agent using Reinforcement learning. Code adapted from the UC Berkeley Pacman Projects (https://inst.eecs.berkeley.edu/~cs188/sp22/projects/)

# Learning Algorithm 

Used the Bellman Update 

<img width="510" alt="Screenshot 2023-12-18 at 1 37 26 AM" src="https://github.com/andrewni420/Pacman-AI/assets/77020120/20330fe2-d63a-4c3a-8943-05b65682d62f">

to learn the expected cumulative discounted reward for each action in a given state

# Reward Shaping

Based the reward function off of potential-based shaping (https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=2352f3c6b22a966c1f483c3e0b376b670a7bf774) to decrease the reward horizon and accelerate training 

# Performance 

Defeats the baseline hand-coded agent in 30/30 games on the defaultCapture layout, with an average score of 10.1 points. 

# Reproducing

To compare the baseline agent to the Q-learning agent over 30 games, run ```python capture.py -r baselineTeam -b myTeam -n 30```
