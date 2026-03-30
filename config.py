from dataclasses import dataclass, field
from pathlib import Path


DEVICE_NAMES = (
    "living_room_light",
    "bedroom_light",
    "kitchen_light",
    "smart_plug_tv",
    "smart_plug_desk_or_coffee",
)

POWER_PROFILE = (1.0, 0.7, 0.9, 1.5, 1.2)


@dataclass(frozen=True)
class PatternRewardWeights:
    habit_match: float = 0.50
    comfort: float = 0.25
    energy: float = 0.15
    switching: float = 0.10


@dataclass(frozen=True)
class PatternConfig:
    num_devices: int = 5
    device_names: tuple[str, ...] = DEVICE_NAMES
    power_profile: tuple[float, ...] = POWER_PROFILE

    step_minutes: int = 15
    steps_per_day: int = 96
    episode_days: int = 7
    episode_steps: int = 672

    state_dim: int = 9
    action_dim: int = 32

    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 100_000
    target_update_steps: int = 1_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000

    hidden_dims: tuple[int, ...] = (128, 128, 64)
    train_episodes: int = 1_000
    seed: int = 7
    device: str = "cpu"

    comfort_dark_threshold: float = 0.35
    comfort_bright_threshold: float = 0.70
    reward_weights: PatternRewardWeights = field(default_factory=PatternRewardWeights)

    save_dir: Path = Path("artifacts")

