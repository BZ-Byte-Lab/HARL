from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from dataclasses import fields, replace
from pathlib import Path
from statistics import mean

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


def format_time(step_in_day: int, step_minutes: int) -> str:
    total_minutes = step_in_day * step_minutes
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def device_labels(device_values: list[int], device_names: tuple[str, ...]) -> str:
    enabled = [name for name, value in zip(device_names, device_values, strict=True) if value]
    return ",".join(enabled) if enabled else "all_off"


def build_config_from_dict(config_dict: dict, device: str) -> PatternConfig:
    valid_fields = {field.name for field in fields(PatternConfig)}
    filtered = {key: value for key, value in config_dict.items() if key in valid_fields}
    filtered["device"] = device
    return PatternConfig(**filtered)


def load_agent(checkpoint_path: Path, device: str) -> tuple[DQNAgent, PatternConfig]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except pickle.UnpicklingError:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config_dict" in checkpoint:
        config = build_config_from_dict(checkpoint["config_dict"], device)
    else:
        saved_config = checkpoint["config"]
        config = replace(saved_config, device=device)

    agent = DQNAgent(config)
    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.total_env_steps = checkpoint.get("total_env_steps", 0)
    agent.policy_net.eval()
    agent.target_net.eval()
    return agent, config


def evaluate_episode(
    env: PatternEnv,
    agent: DQNAgent,
    seed: int,
    show_trace: bool,
    trace_limit: int | None,
) -> dict:
    state, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    total_matches = 0
    total_switches = 0
    total_steps = 0
    day_rewards = defaultdict(float)
    trace_rows: list[str] = []

    while not done:
        action = agent.select_action(state, greedy=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        total_steps += 1
        day_rewards[info["day_index"]] += reward
        total_switches += sum(env.device_state[idx] != env.previous_device_state[idx] for idx in range(env.config.num_devices))

        matches = sum(
            predicted == ideal
            for predicted, ideal in zip(env.device_state, info["ideal_devices"], strict=True)
        )
        total_matches += matches

        if show_trace and (trace_limit is None or len(trace_rows) < trace_limit):
            trace_rows.append(
                "day=%d time=%s home=%d light=%.2f action=%02d selected=%s ideal=%s reward=%.3f"
                % (
                    info["day_index"],
                    format_time(info["step_in_day"], env.config.step_minutes),
                    info["is_home"],
                    info["light_level"],
                    action,
                    device_labels(env.device_state, env.config.device_names),
                    device_labels(info["ideal_devices"], env.config.device_names),
                    reward,
                )
            )

        state = next_state

    return {
        "episode_reward": total_reward,
        "match_rate": total_matches / (total_steps * env.config.num_devices),
        "avg_switches_per_step": total_switches / total_steps,
        "day_rewards": dict(day_rewards),
        "trace_rows": trace_rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained Pattern Agent checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/pattern_dqn.pt"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--show-trace", action="store_true")
    parser.add_argument("--trace-limit", type=int, default=40)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = args.device or detect_device()
    agent, config = load_agent(args.checkpoint, device)
    env = PatternEnv(config)

    results = [
        evaluate_episode(
            env=env,
            agent=agent,
            seed=args.seed + episode_index,
            show_trace=args.show_trace,
            trace_limit=args.trace_limit,
        )
        for episode_index in range(args.episodes)
    ]

    print(f"checkpoint={args.checkpoint}")
    print(f"device={device}")
    print(f"episodes={args.episodes}")
    print(f"mean_episode_reward={mean(result['episode_reward'] for result in results):.3f}")
    print(f"mean_match_rate={mean(result['match_rate'] for result in results):.3%}")
    print(
        "mean_switches_per_step=%.3f"
        % mean(result["avg_switches_per_step"] for result in results)
    )

    for episode_index, result in enumerate(results, start=1):
        print(
            "episode=%d reward=%.3f match_rate=%.3f switches_per_step=%.3f"
            % (
                episode_index,
                result["episode_reward"],
                result["match_rate"],
                result["avg_switches_per_step"],
            )
        )
        print(f"episode={episode_index} day_rewards={result['day_rewards']}")
        if args.show_trace and result["trace_rows"]:
            print(f"episode={episode_index} trace:")
            for row in result["trace_rows"]:
                print(row)


if __name__ == "__main__":
    main()
