from __future__ import annotations

import argparse
import random
from dataclasses import replace
from statistics import mean

import numpy as np
import torch

from config import PatternConfig
from dqn_agent import DQNAgent
from pattern_env import PatternEnv


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_policy(env: PatternEnv, agent: DQNAgent, episodes: int, seed: int) -> float:
    scores: list[float] = []
    for episode_index in range(episodes):
        state, _ = env.reset(seed=seed + 10_000 + episode_index)
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.select_action(state, greedy=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward
            done = terminated or truncated

        scores.append(episode_reward)

    return mean(scores)


def train(config: PatternConfig) -> None:
    set_seed(config.seed)

    env = PatternEnv(config)
    agent = DQNAgent(config)
    reward_history: list[float] = []

    for episode_index in range(config.train_episodes):
        state, _ = env.reset(seed=config.seed + episode_index)
        done = False
        episode_reward = 0.0
        episode_losses: list[float] = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.observe(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            episode_reward += reward

        reward_history.append(episode_reward)

        if (episode_index + 1) % 10 == 0:
            avg_reward = mean(reward_history[-10:])
            avg_loss = mean(episode_losses) if episode_losses else 0.0
            print(
                f"episode={episode_index + 1} "
                f"avg_reward_10={avg_reward:.3f} "
                f"avg_loss={avg_loss:.4f} "
                f"epsilon={agent.epsilon:.3f}"
            )

    checkpoint_path = config.save_dir / "pattern_dqn.pt"
    agent.save(checkpoint_path)
    eval_reward = evaluate_policy(env, agent, episodes=5, seed=config.seed)

    print(f"saved_checkpoint={checkpoint_path}")
    print(f"greedy_eval_reward={eval_reward:.3f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the minimal Pattern Agent DQN.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = replace(
        PatternConfig(),
        train_episodes=args.episodes,
        seed=args.seed,
        device=args.device or detect_device(),
    )
    train(config)


if __name__ == "__main__":
    main()
