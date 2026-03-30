from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import torch
from torch import nn, optim

from config import PatternConfig


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)
        return states, actions, rewards, next_states, dones


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNAgent:
    def __init__(self, config: PatternConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.policy_net = QNetwork(
            config.state_dim, config.action_dim, config.hidden_dims
        ).to(self.device)
        self.target_net = QNetwork(
            config.state_dim, config.action_dim, config.hidden_dims
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.replay_buffer = ReplayBuffer(config.replay_capacity)
        self.total_env_steps = 0

    def select_action(self, state, greedy: bool = False) -> int:
        epsilon = 0.0 if greedy else self.epsilon
        if random.random() < epsilon:
            return random.randrange(self.config.action_dim)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def observe(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.total_env_steps += 1

    def train_step(self) -> float | None:
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )

        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_tensor = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.policy_net(states_tensor).gather(1, actions_tensor)
        with torch.no_grad():
            next_q = self.target_net(next_states_tensor).max(dim=1, keepdim=True).values
            target_q = rewards_tensor + self.config.gamma * (1.0 - dones_tensor) * next_q

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.total_env_steps % self.config.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.total_env_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def save(self, path: str | Path) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_env_steps": self.total_env_steps,
                "config": self.config,
            },
            save_path,
        )

