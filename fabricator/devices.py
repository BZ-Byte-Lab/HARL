from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    category: str
    kind: str
    power_watts: float | None = None


LIGHT_NAMES = (
    "kitchen_light",
    "bedroom_light",
    "living_room_light",
    "bathroom_light",
    "hallway_light",
    "office_light",
    "dining_room_light",
    "bedside_lamp",
    "porch_light",
)

APPLIANCE_NAMES = (
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
)

PRESENCE_NAMES = (
    "person_home",
    "front_door",
    "motion_kitchen",
    "motion_bedroom",
    "motion_living_room",
    "motion_bathroom",
    "motion_hallway",
    "motion_office",
)

AMBIENT_TIME_NAMES = (
    "ambient_light_outdoor",
    "ambient_light_bedroom",
    "indoor_temperature_c",
    "timestamp",
    "time_of_day_norm",
    "day_of_week",
    "is_weekend",
)

DEVICES = (
    *(DeviceSpec(name=name, category="lights", kind="binary", power_watts=8.0) for name in LIGHT_NAMES),
    *(DeviceSpec(name=name, category="appliances", kind="binary") for name in APPLIANCE_NAMES),
    *(DeviceSpec(name=name, category="presence", kind="binary") for name in PRESENCE_NAMES),
    DeviceSpec(name="ambient_light_outdoor", category="ambient_time", kind="continuous"),
    DeviceSpec(name="ambient_light_bedroom", category="ambient_time", kind="continuous"),
    DeviceSpec(name="indoor_temperature_c", category="ambient_time", kind="continuous"),
    DeviceSpec(name="timestamp", category="ambient_time", kind="metadata"),
    DeviceSpec(name="time_of_day_norm", category="ambient_time", kind="continuous"),
    DeviceSpec(name="day_of_week", category="ambient_time", kind="metadata"),
    DeviceSpec(name="is_weekend", category="ambient_time", kind="metadata"),
)


def devices_by_category() -> dict[str, tuple[DeviceSpec, ...]]:
    grouped: dict[str, list[DeviceSpec]] = defaultdict(list)
    for device in DEVICES:
        grouped[device.category].append(device)
    return {category: tuple(items) for category, items in grouped.items()}
