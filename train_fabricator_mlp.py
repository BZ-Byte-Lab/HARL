from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


NUMERIC_CONTEXT_FEATURES = (
    "time_of_day_norm",
    "day_of_week",
    "is_weekend",
    "sunrise_minute",
    "sunset_minute",
    "cloudiness",
)

TARGET_COLUMNS = (
    "kitchen_light",
    "bedroom_light",
    "living_room_light",
    "bathroom_light",
    "hallway_light",
    "office_light",
    "dining_room_light",
    "bedside_lamp",
    "porch_light",
    "tv_living_room",
    "tv_bedroom",
    "coffee_maker_plug",
    "kettle_plug",
    "microwave_plug",
    "desk_monitor_plug",
    "game_console_plug",
    "washing_machine_plug",
    "dryer_plug",
    "dishwasher_plug",
    "robot_vacuum_plug",
    "space_heater_plug",
    "person_home",
    "front_door",
    "motion_kitchen",
    "motion_bedroom",
    "motion_living_room",
    "motion_bathroom",
    "motion_hallway",
    "motion_office",
    "ambient_light_outdoor",
    "ambient_light_bedroom",
    "indoor_temperature_c",
)

BINARY_TARGETS = TARGET_COLUMNS[:-3]
CONTINUOUS_TARGETS = TARGET_COLUMNS[-3:]


class SimulatorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a minimal MLP simulator on fabricated smart-home CSV data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("artifacts/fabricated"),
        help="Root directory containing fabricated persona CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/fabricator_mlp_simulator.pt"),
        help="Checkpoint path.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(128, 128))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, e.g. cpu, cuda, mps.",
    )
    return parser.parse_args()


def safe_float(value: str) -> float:
    lowered = value.lower()
    if lowered == "nan":
        return 0.0
    if lowered == "true":
        return 1.0
    if lowered == "false":
        return 0.0
    return float(value)


def discover_csv_files(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.rglob("*.csv") if path.is_file())


def load_rows(csv_paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in csv_paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(reader)
    if not rows:
        raise ValueError("No rows loaded from CSV files.")
    return rows


def collect_categories(rows: list[dict[str, str]], column: str) -> list[str]:
    return sorted({row[column] for row in rows})


def normalize_context(row: dict[str, str]) -> list[float]:
    return [
        safe_float(row["time_of_day_norm"]),
        safe_float(row["day_of_week"]) / 6.0,
        safe_float(row["is_weekend"]),
        safe_float(row["sunrise_minute"]) / 1440.0,
        safe_float(row["sunset_minute"]) / 1440.0,
        safe_float(row["cloudiness"]),
    ]


def target_vector(row: dict[str, str]) -> list[float]:
    vector = [safe_float(row[name]) for name in TARGET_COLUMNS]
    vector[-1] = vector[-1] / 40.0
    return vector


def one_hot(value: str, vocabulary: list[str]) -> list[float]:
    return [1.0 if item == value else 0.0 for item in vocabulary]


def build_examples(
    rows: list[dict[str, str]],
    persona_vocab: list[str],
    season_vocab: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[list[float]] = []
    targets: list[list[float]] = []
    previous_target_by_stream: dict[tuple[str, str], list[float]] = {}

    for row in rows:
        stream_key = (row["persona_label"], row["timestamp"][:10])
        previous_target = previous_target_by_stream.get(stream_key)
        if previous_target is None:
            previous_target = [0.0] * len(TARGET_COLUMNS)

        feature_vector = [
            *normalize_context(row),
            *one_hot(row["persona_label"], persona_vocab),
            *one_hot(row["season_label"], season_vocab),
            *previous_target,
        ]
        current_target = target_vector(row)
        features.append(feature_vector)
        targets.append(current_target)
        previous_target_by_stream[stream_key] = current_target

    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(targets, dtype=torch.float32)
    return x_tensor, y_tensor


def split_outputs(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, ...]:
    binary_dim = len(BINARY_TARGETS)
    pred_binary = predictions[:, :binary_dim]
    pred_continuous = predictions[:, binary_dim:]
    target_binary = targets[:, :binary_dim]
    target_continuous = targets[:, binary_dim:]
    return pred_binary, pred_continuous, target_binary, target_continuous


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_binary_correct = 0.0
    total_binary_count = 0
    loss_fn_binary = nn.BCEWithLogitsLoss()
    loss_fn_continuous = nn.MSELoss()

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features)
            pred_binary, pred_continuous, target_binary, target_continuous = split_outputs(
                predictions,
                targets,
            )
            loss = loss_fn_binary(pred_binary, target_binary) + loss_fn_continuous(
                pred_continuous,
                target_continuous,
            )
            total_loss += float(loss.item()) * len(features)

            binary_probs = torch.sigmoid(pred_binary)
            binary_preds = (binary_probs >= 0.5).float()
            total_binary_correct += float((binary_preds == target_binary).sum().item())
            total_binary_count += target_binary.numel()

    mean_loss = total_loss / max(1, len(data_loader.dataset))
    binary_accuracy = total_binary_correct / max(1, total_binary_count)
    return mean_loss, binary_accuracy


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    csv_paths = discover_csv_files(args.data_dir)
    if not csv_paths:
        raise SystemExit(
            f"No CSV files found under {args.data_dir}. Generate data first with "
            "'python -m fabricator.cli generate-all --weeks 4 --seed 42 --format csv'."
        )

    rows = load_rows(csv_paths)
    persona_vocab = collect_categories(rows, "persona_label")
    season_vocab = collect_categories(rows, "season_label")
    features, targets = build_examples(rows, persona_vocab, season_vocab)

    dataset = TensorDataset(features, targets)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = SimulatorMLP(
        input_dim=features.shape[1],
        hidden_dims=tuple(args.hidden_dims),
        output_dim=targets.shape[1],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn_binary = nn.BCEWithLogitsLoss()
    loss_fn_continuous = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            predictions = model(batch_features)
            pred_binary, pred_continuous, target_binary, target_continuous = split_outputs(
                predictions,
                batch_targets,
            )
            loss = loss_fn_binary(pred_binary, target_binary) + loss_fn_continuous(
                pred_continuous,
                target_continuous,
            )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_features)

        train_loss = running_loss / max(1, len(train_loader.dataset))
        val_loss, val_binary_accuracy = evaluate(model, val_loader, device)
        print(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_binary_accuracy=%.4f"
            % (epoch, train_loss, val_loss, val_binary_accuracy)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": features.shape[1],
        "output_dim": targets.shape[1],
        "hidden_dims": tuple(args.hidden_dims),
        "persona_vocab": persona_vocab,
        "season_vocab": season_vocab,
        "numeric_context_features": NUMERIC_CONTEXT_FEATURES,
        "target_columns": TARGET_COLUMNS,
        "binary_targets": BINARY_TARGETS,
        "continuous_targets": CONTINUOUS_TARGETS,
    }
    torch.save(checkpoint, args.output)

    print(f"samples={len(dataset)}")
    print(f"train_samples={len(train_dataset)}")
    print(f"val_samples={len(val_dataset)}")
    print(f"checkpoint={args.output}")


if __name__ == "__main__":
    main()
