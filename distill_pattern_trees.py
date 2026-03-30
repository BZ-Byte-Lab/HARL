from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree, export_text

from config import PatternConfig
from evaluate_pattern import detect_device, load_agent
from pattern_env import PatternEnv
from world_state import action_to_device_vector


FEATURE_NAMES = (
    "time_of_day",
    "day_of_week",
    "is_home",
    "light_level",
    "living_room_light_current",
    "bedroom_light_current",
    "kitchen_light_current",
    "smart_plug_tv_current",
    "smart_plug_desk_or_coffee_current",
)


def humanize_feature(feature_name: str) -> str:
    mapping = {
        "time_of_day": "time of day",
        "day_of_week": "day of week",
        "is_home": "occupancy state",
        "light_level": "ambient light level",
        "living_room_light_current": "current living room light state",
        "bedroom_light_current": "current bedroom light state",
        "kitchen_light_current": "current kitchen light state",
        "smart_plug_tv_current": "current TV plug state",
        "smart_plug_desk_or_coffee_current": "current desk or coffee plug state",
    }
    return mapping.get(feature_name, feature_name)


def format_threshold(feature_name: str, threshold: float) -> str:
    if feature_name == "time_of_day":
        total_minutes = int(round(float(threshold) * 24 * 60))
        total_minutes = max(0, min(24 * 60 - 1, total_minutes))
        hour = total_minutes // 60
        minute = total_minutes % 60
        return f"{hour:02d}:{minute:02d}"
    if feature_name == "day_of_week":
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        index = int(round(float(threshold) * 6))
        index = max(0, min(6, index))
        return days[index]
    if feature_name in {"is_home"} or feature_name.endswith("_current"):
        return "on / home" if threshold >= 0.5 else "off / away"
    return f"{threshold:.3f}"


def collect_policy_dataset(
    checkpoint_path: Path,
    episodes: int,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, PatternConfig]:
    agent, config = load_agent(checkpoint_path, device)
    env = PatternEnv(config)

    states: list[np.ndarray] = []
    actions: list[int] = []

    for episode_index in range(episodes):
        state, _ = env.reset(seed=seed + episode_index)
        done = False
        while not done:
            action = agent.select_action(state, greedy=True)
            states.append(np.asarray(state, dtype=np.float32))
            actions.append(action)
            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state
            done = terminated or truncated

    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.int64), config


def tree_to_natural_language(
    tree: DecisionTreeClassifier,
    feature_names: tuple[str, ...],
    device_name: str,
) -> list[str]:
    tree_ = tree.tree_
    rules: list[str] = []

    def walk(node_id: int, conditions: list[str]) -> None:
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
            feature_name = feature_names[tree_.feature[node_id]]
            threshold = tree_.threshold[node_id]

            left_condition = f"{humanize_feature(feature_name)} <= {format_threshold(feature_name, threshold)}"
            right_condition = f"{humanize_feature(feature_name)} > {format_threshold(feature_name, threshold)}"

            walk(tree_.children_left[node_id], conditions + [left_condition])
            walk(tree_.children_right[node_id], conditions + [right_condition])
            return

        value = tree_.value[node_id][0]
        predicted_class = int(np.argmax(value))
        sample_count = int(np.sum(value))
        action_text = "turn on" if predicted_class == 1 else "turn off"
        condition_text = " and ".join(conditions) if conditions else "any state"
        rules.append(
            f"If {condition_text}, then {action_text} {device_name}. sample_count={sample_count}"
        )

    walk(0, [])
    return rules


def train_device_tree(
    features: np.ndarray,
    labels: np.ndarray,
    max_depth: int,
    seed: int,
) -> DecisionTreeClassifier:
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=seed,
        min_samples_leaf=20,
    )
    tree.fit(features, labels)
    return tree


def write_tree_report(
    output_dir: Path,
    device_name: str,
    tree: DecisionTreeClassifier,
    accuracy: float,
    positive_rate: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{device_name}_decision_tree.txt"
    readable_tree = export_text(tree, feature_names=list(FEATURE_NAMES), decimals=3)
    natural_rules = tree_to_natural_language(tree, FEATURE_NAMES, device_name)

    lines = [
        f"Device: {device_name}",
        f"Training-set accuracy: {accuracy:.4f}",
        f"Positive rate in distilled labels: {positive_rate:.4f}",
        f"Tree depth: {tree.get_depth()}",
        f"Leaf count: {tree.get_n_leaves()}",
        "",
        "Structured tree text:",
        readable_tree,
        "",
        "Natural-language rules:",
        *natural_rules,
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distill one max-depth-8 decision tree per device from the Pattern RL policy."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/pattern_dqn.pt"))
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/decision_trees"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    runtime_device = args.device or detect_device()

    features, actions, config = collect_policy_dataset(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        device=runtime_device,
    )

    action_vectors = np.asarray(
        [action_to_device_vector(int(action), config.num_devices) for action in actions],
        dtype=np.int64,
    )

    print(f"checkpoint={args.checkpoint}")
    print(f"runtime_device={runtime_device}")
    print(f"distillation_episodes={args.episodes}")
    print(f"samples={len(features)}")
    print(f"action_distribution={dict(sorted(Counter(actions).items()))}")

    summary_lines = []
    for device_index, device_name in enumerate(config.device_names):
        labels = action_vectors[:, device_index]
        tree = train_device_tree(
            features=features,
            labels=labels,
            max_depth=args.max_depth,
            seed=args.seed,
        )
        predictions = tree.predict(features)
        accuracy = float(np.mean(predictions == labels))
        positive_rate = float(np.mean(labels))
        write_tree_report(
            output_dir=args.output_dir,
            device_name=device_name,
            tree=tree,
            accuracy=accuracy,
            positive_rate=positive_rate,
        )
        summary_lines.append(
            "device=%s accuracy=%.4f positive_rate=%.4f depth=%d leaves=%d"
            % (
                device_name,
                accuracy,
                positive_rate,
                tree.get_depth(),
                tree.get_n_leaves(),
            )
        )

    summary_path = args.output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("decision_tree_reports:")
    for line in summary_lines:
        print(line)
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()
