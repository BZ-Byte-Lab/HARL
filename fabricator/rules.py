from __future__ import annotations

import math
import random
from dataclasses import dataclass

from fabricator.ambient import per_room_ambient
from fabricator.personas import Persona
from fabricator.schedule import DayPlan


@dataclass(frozen=True)
class AnomalyConfig:
    forgot_light_off_prob: float = 0.02
    sensor_dropout_prob: float = 0.01
    phantom_motion_prob: float = 0.01


def _window_active(minute: int, start: int | None, duration: int) -> bool:
    if start is None:
        return False
    return start <= minute < start + duration


def _range_active(minute: int, start: int | None, end: int | None) -> bool:
    if start is None or end is None:
        return False
    if start <= end:
        return start <= minute < end
    return minute >= start or minute < end


def _work_window(plan: DayPlan) -> tuple[int, int]:
    if plan.leave_home_minute is not None and plan.return_home_minute is not None:
        return plan.leave_home_minute, plan.return_home_minute
    return 9 * 60, 17 * 60


def _is_sleeping(plan: DayPlan, minute: int) -> bool:
    if plan.wake_minute <= plan.sleep_minute:
        return minute < plan.wake_minute or minute >= plan.sleep_minute
    return plan.sleep_minute <= minute < plan.wake_minute


def _active_room(
    persona: Persona,
    plan: DayPlan,
    minute: int,
    is_home: bool,
) -> str | None:
    if not is_home:
        return None
    if _is_sleeping(plan, minute):
        return None
    if _window_active(minute, plan.wake_minute, 15) or _window_active(
        minute,
        plan.sleep_minute - 30,
        30,
    ):
        return "bathroom"
    if _window_active(minute, plan.breakfast_minute, 30) or _window_active(minute, plan.dinner_minute, 45):
        return "kitchen"
    if _range_active(minute, plan.tv_start_minute, plan.tv_end_minute):
        return "bedroom" if persona.primary_daytime_room == "bedroom" else "living_room"
    if _window_active(minute, plan.wind_down_minute, 60):
        return "bedroom"
    if persona.primary_daytime_room == "office":
        start, end = _work_window(plan)
        if start <= minute < end and plan.work_mode == "home":
            return "office"
    if persona.primary_daytime_room == "living_room":
        return "living_room"
    if persona.primary_daytime_room == "bedroom":
        return "bedroom"
    return "living_room" if minute >= 18 * 60 else "kitchen"


def compute_lights(
    persona: Persona,
    plan: DayPlan,
    minute: int,
    is_home: bool,
    outdoor_light: float,
) -> dict[str, int]:
    active_room = _active_room(persona, plan, minute, is_home)
    lights = {
        "kitchen_light": int(
            is_home
            and (
                _window_active(minute, plan.breakfast_minute, 30)
                or _window_active(minute, plan.lunch_minute, 30)
                or _window_active(minute, plan.dinner_minute, 45)
            )
        ),
        "bedroom_light": int(is_home and active_room == "bedroom" and outdoor_light < 0.65),
        "living_room_light": int(
            is_home and active_room == "living_room" and (outdoor_light < 0.55 or _range_active(minute, plan.tv_start_minute, plan.tv_end_minute))
        ),
        "bathroom_light": int(
            is_home
            and (
                _window_active(minute, plan.wake_minute, 15)
                or _window_active(minute, plan.leave_home_minute, 15)
                or _window_active(minute, plan.sleep_minute - 15, 15)
            )
        ),
        "hallway_light": int(
            any(abs(minute - anchor) < 15 for anchor in (plan.leave_home_minute, plan.return_home_minute) if anchor is not None)
        ),
        "office_light": int(
            is_home
            and persona.primary_daytime_room == "office"
            and plan.work_mode == "home"
            and 9 * 60 <= minute < 17 * 60
            and outdoor_light < 0.75
        ),
        "dining_room_light": int(is_home and _window_active(minute, plan.dinner_minute, 45) and outdoor_light < 0.7),
        "bedside_lamp": int(is_home and _window_active(minute, plan.wind_down_minute, 45)),
        "porch_light": int(outdoor_light < 0.2 and (is_home or persona.name in {"commuter", "social"})),
    }

    if outdoor_light > 0.9:
        for name in ("kitchen_light", "living_room_light", "office_light", "dining_room_light"):
            lights[name] = 0
    return lights


def compute_plugs(
    persona: Persona,
    plan: DayPlan,
    minute: int,
    is_home: bool,
    appliance_events: dict[str, list[tuple[int, int]]],
    lights: dict[str, int] | None = None,
    outdoor_temp: float | None = None,
) -> dict[str, int]:
    lights = lights or {}
    tv_room = "tv_bedroom" if persona.primary_daytime_room == "bedroom" else "tv_living_room"
    plugs = {
        "tv_living_room": int(tv_room == "tv_living_room" and is_home and _range_active(minute, plan.tv_start_minute, plan.tv_end_minute)),
        "tv_bedroom": int(tv_room == "tv_bedroom" and is_home and _range_active(minute, plan.tv_start_minute, plan.tv_end_minute)),
        "coffee_maker_plug": int(is_home and abs(minute - (plan.wake_minute + 15)) < 15),
        "kettle_plug": int(is_home and persona.name in {"wfh", "hybrid", "retiree"} and _window_active(minute, 15 * 60, 15)),
        "microwave_plug": int(
            is_home and (
                _window_active(minute, plan.lunch_minute, 15)
                or _window_active(minute, plan.dinner_minute, 15)
            )
        ),
        "desk_monitor_plug": int(
            is_home
            and persona.primary_daytime_room == "office"
            and plan.work_mode == "home"
            and 9 * 60 <= minute < 17 * 60
            and bool(lights.get("office_light"))
        ),
        "game_console_plug": int(
            is_home
            and persona.name in {"student", "social"}
            and (
                _window_active(minute, 20 * 60, 60)
                or (_window_active(minute, 22 * 60, 90) and persona.name == "student")
            )
        ),
        "washing_machine_plug": 0,
        "dryer_plug": 0,
        "dishwasher_plug": 0,
        "robot_vacuum_plug": 0,
        "space_heater_plug": int(
            is_home
            and outdoor_temp is not None
            and outdoor_temp < 5.0
            and (_window_active(minute, plan.wake_minute, 30) or _window_active(minute, plan.wind_down_minute, 30))
        ),
    }

    for name, windows in appliance_events.items():
        if any(start <= minute < end for start, end in windows):
            plugs[name] = 1

    return plugs


def compute_presence(
    persona: Persona,
    plan: DayPlan,
    minute: int,
    is_home: bool,
    away_blocks: list[tuple[int, int]],
    lights: dict[str, int] | None = None,
    rng: random.Random | None = None,
) -> dict[str, int]:
    lights = lights or {}
    active_room = _active_room(persona, plan, minute, is_home)

    front_door = 0
    for start, end in away_blocks:
        if abs(minute - start) < 15 or abs(minute - end) < 15:
            front_door = 1
            break

    motions = {
        "motion_kitchen": 0,
        "motion_bedroom": 0,
        "motion_living_room": 0,
        "motion_bathroom": 0,
        "motion_hallway": int(front_door or lights.get("hallway_light", 0)),
        "motion_office": 0,
    }

    room_to_sensor = {
        "kitchen": "motion_kitchen",
        "bedroom": "motion_bedroom",
        "living_room": "motion_living_room",
        "bathroom": "motion_bathroom",
        "office": "motion_office",
    }
    if is_home and active_room in room_to_sensor:
        motions[room_to_sensor[active_room]] = 1

    if rng is not None and is_home:
        for sensor_name, light_name in (
            ("motion_kitchen", "kitchen_light"),
            ("motion_bedroom", "bedroom_light"),
            ("motion_living_room", "living_room_light"),
            ("motion_bathroom", "bathroom_light"),
            ("motion_office", "office_light"),
        ):
            if lights.get(light_name, 0) and rng.random() < 0.8:
                motions[sensor_name] = 1

    return {
        "person_home": int(is_home),
        "front_door": int(front_door),
        **motions,
    }


def compute_ambient(
    plan: DayPlan,
    minute: int,
    outdoor_light: float,
    indoor_temp_c: float,
    curtain_factor: float,
) -> dict[str, float]:
    return {
        "ambient_light_outdoor": outdoor_light,
        "ambient_light_bedroom": per_room_ambient(outdoor_light, curtain_factor),
        "indoor_temperature_c": indoor_temp_c,
        "time_of_day_norm": minute / 1440.0,
        "day_of_week": float(plan.day_of_week),
        "is_weekend": float(int(plan.is_weekend)),
    }


def apply_anomalies(
    snapshot: dict[str, float | int | str],
    persona: Persona,
    plan: DayPlan,
    minute: int,
    is_home: bool,
    rng: random.Random,
    config: AnomalyConfig,
) -> None:
    if minute >= plan.wind_down_minute and rng.random() < config.forgot_light_off_prob:
        target = "bedroom_light" if persona.primary_daytime_room == "bedroom" else "living_room_light"
        snapshot[target] = 1.0

    if is_home and rng.random() < config.phantom_motion_prob:
        snapshot[rng.choice(["motion_kitchen", "motion_bedroom", "motion_living_room", "motion_office"])] = 1.0

    if rng.random() < config.sensor_dropout_prob:
        snapshot[rng.choice(
            [
                "ambient_light_outdoor",
                "ambient_light_bedroom",
                "motion_kitchen",
                "motion_bedroom",
                "motion_living_room",
                "motion_bathroom",
                "motion_hallway",
                "motion_office",
                "front_door",
            ]
        )] = math.nan
