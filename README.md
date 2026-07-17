# Connect4-RL

## Overview

**Connect4-RL** is a research-oriented repository that explores how **Reinforcement Learning (RL)** algorithms can learn to play **Connect Four**, a classic two-player, zero-sum game with perfect information.

The primary goal of this project is **not** to rediscover the mathematically optimal strategy for Connect Four. Instead, the objective is to investigate how reinforcement learning agents can learn effective strategies through self-play and interaction with the environment, providing insights into **multi-agent reinforcement learning (MARL)**.

This repository serves as a platform for experimenting with RL algorithms, comparing learning methods, and studying emergent behaviors in competitive environments.

---

# Why Connect Four?

Connect Four is one of the most popular benchmark games in artificial intelligence because it possesses several desirable properties:

- Two-player adversarial environment
- Zero-sum game
- Perfect information
- Deterministic dynamics
- Large but manageable state space
- Clear win/loss objective

Unlike simple games such as Tic-Tac-Toe, Connect Four is sufficiently complex to require long-term planning while remaining computationally feasible for experimentation.

Because of these characteristics, Connect Four has become a standard testbed for search algorithms, game theory, and reinforcement learning.

---

# Connect Four is a Solved Game

One of the most interesting aspects of Connect Four is that it is a **solved game**.

In game theory, a game is considered **solved** when its optimal outcome is known assuming both players always make perfect decisions.

For the standard 7×6 Connect Four board:

> **The first player can always force a win with perfect play by opening in the center column.**

This result was independently proven in **1988** by **James Dow Allen** and **Victor Allis** using game-theoretic analysis and extensive search techniques. Later work verified the result using increasingly efficient search algorithms and endgame databases. :contentReference[oaicite:0]{index=0}

It is important to understand what "solved" means.

It **does not** mean that the strategy is simple or easy for humans to memorize.

Instead, it means that there exists an optimal policy that guarantees the theoretical outcome regardless of the opponent's actions.

Finding this policy required exploring an enormous game tree containing trillions of possible positions, making Connect Four an important milestone in computational game theory. :contentReference[oaicite:1]{index=1}

---

# Why Study a Solved Game?

At first glance, studying a solved game may seem unnecessary.

However, solved games are among the best environments for reinforcement learning research because they provide a **known optimal solution** against which learning algorithms can be evaluated.

Using Connect Four allows researchers to answer questions such as:

- Can an RL agent discover strong strategies without human knowledge?
- How efficiently can different algorithms learn?
- How much exploration is required?
- Can neural networks approximate optimal play?
- How well does self-play scale?
- How closely does a learned policy approach the theoretical optimum?

Because the optimal outcome is already known, performance can be measured objectively rather than relying solely on empirical win rates.

---

# Reinforcement Learning and Multi-Agent Problems

Most introductory reinforcement learning problems involve a **single agent** interacting with a stationary environment.

Examples include:

- CartPole
- MountainCar
- LunarLander

In these problems, the environment does not actively change its strategy.

Competitive games are fundamentally different.

In Connect Four, the environment is another intelligent agent whose actions directly influence future states.

This creates several additional challenges:

- The environment becomes **non-stationary** because the opponent's behavior changes over time.
- Rewards are delayed until the end of the game.
- Small mistakes can have consequences many moves later.
- Exploration becomes significantly more difficult.
- Learning stability is harder to achieve.

These characteristics make Connect Four an excellent benchmark for **Multi-Agent Reinforcement Learning (MARL)**.

---

# Why Multi-Agent Reinforcement Learning Matters

Many real-world decision-making problems involve multiple intelligent agents rather than a single learner.

Examples include:

- Autonomous driving
- Robot coordination
- Traffic control
- Financial markets
- Cybersecurity
- Resource allocation
- Strategic planning
- Multi-robot systems
- Competitive games

In these domains, every agent influences the environment experienced by the others.

Traditional reinforcement learning algorithms often assume that the environment remains stationary, an assumption that breaks down in multi-agent settings.

Developing algorithms capable of learning robust policies in competitive and cooperative environments remains one of the central research challenges in modern reinforcement learning.

Game-playing environments like Connect Four provide a controlled setting for studying these problems before applying similar techniques to more complex real-world systems.

---

# Project Goals

This repository aims to:

- Implement Connect Four as an RL environment.
- Train reinforcement learning agents through self-play.
- Compare different reinforcement learning algorithms.
- Study convergence toward optimal strategies.
- Investigate exploration techniques.
- Evaluate learned policies against strong opponents.
- Analyze learning dynamics in a multi-agent setting.

The long-term objective is to better understand how reinforcement learning algorithms behave in adversarial environments and how they can scale to increasingly complex multi-agent problems.

---

# Future Directions

Potential future extensions include:

- Deep Q-Networks (DQN)
- Double DQN
- Dueling Networks
- Prioritized Experience Replay
- Policy Gradient methods
- Actor-Critic algorithms
- Proximal Policy Optimization (PPO)
- Monte Carlo Tree Search (MCTS)
- AlphaZero-style self-play
- Population-based training
- Curriculum learning

---

# Future changes
This repository is incomplete. Since I am researching in this area, I might update it later.

# References
- Victor Allis. *A Knowledge-based Approach of Connect-Four*. Master's Thesis, Vrije Universiteit Amsterdam, 1988. :contentReference[oaicite:2]{index=2}
- James Dow Allen. *Expert Play in Connect-Four*. 1990. :contentReference[oaicite:3]{index=3}
- Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction (2nd Edition).*
