from __future__ import annotations

import random
from dataclasses import dataclass

from fabricator.personas import Persona, TimeDist


@dataclass(frozen=True)
class DayPlan:
    day_index: int
    day_of_week: int
    is_weekend: bool
    work_mode: str
    wake_minute: int
    leave_home_minute: int | None
    return_home_minute: int | None
    wind_down_minute: int
    sleep_minute: int
    breakfast_minute: int | None
    lunch_minute: int | None
    dinner_minute: int
    tv_start_minute: int | None
    tv_end_minute: int | None
    sunrise_minute: int
    sunset_minute: int
    cloudiness: float


def jittered_minute(rng: random.Random, dist: TimeDist) -> int:
    jittered = round(rng.gauss(dist.mean, dist.stddev))
    return max(dist.min, min(dist.max, jittered))


def _shifted_dist(dist: TimeDist | None, shift_minutes: int) -> TimeDist | None:
    if dist is None:
        return None
    return TimeDist(
        mean=max(0, min(1439, dist.mean + shift_minutes)),
        stddev=dist.stddev,
        min=max(0, min(1439, dist.min + shift_minutes)),
        max=max(0, min(1439, dist.max + shift_minutes)),
    )


def sample_day_schedule(rng: random.Random, persona: Persona, day_index: int) -> DayPlan:
    day_of_week = day_index % 7
    is_weekend = day_of_week >= 5
    shift = persona.weekend_shift_minutes if is_weekend else 0

    wake = jittered_minute(rng, _shifted_dist(persona.wake, shift))
    wind_down = jittered_minute(rng, _shifted_dist(persona.wind_down, shift))
    sleep = jittered_minute(rng, _shifted_dist(persona.sleep, shift))
    breakfast = (
        jittered_minute(rng, _shifted_dist(persona.breakfast, shift))
        if persona.breakfast is not None
        else None
    )
    lunch = (
        jittered_minute(rng, _shifted_dist(persona.lunch, shift))
        if persona.lunch is not None
        else None
    )
    dinner = jittered_minute(rng, _shifted_dist(persona.dinner, shift))

    work_mode = "home"
    leave_home = None
    return_home = None
    if persona.name == "hybrid" and not is_weekend:
        work_mode = "away" if rng.random() < 0.45 else "home"
    elif persona.name in {"commuter", "early_shift", "social", "traveler"} and not is_weekend:
        work_mode = "away"
    elif persona.name == "night_shift" and day_of_week < 5:
        work_mode = "away"

    if work_mode == "away" and persona.leave_home is not None and persona.return_home is not None:
        leave_home = jittered_minute(rng, _shifted_dist(persona.leave_home, shift))
        return_home = jittered_minute(rng, _shifted_dist(persona.return_home, shift))

    tv_start = (
        jittered_minute(rng, _shifted_dist(persona.tv_start, shift))
        if persona.tv_start is not None
        else None
    )
    tv_end = None
    if tv_start is not None:
        duration = rng.randint(*persona.tv_duration_minutes)
        tv_end = (tv_start + duration) % 1440

    return DayPlan(
        day_index=day_index,
        day_of_week=day_of_week,
        is_weekend=is_weekend,
        work_mode=work_mode,
        wake_minute=wake,
        leave_home_minute=leave_home,
        return_home_minute=return_home,
        wind_down_minute=wind_down,
        sleep_minute=sleep,
        breakfast_minute=breakfast,
        lunch_minute=lunch,
        dinner_minute=dinner,
        tv_start_minute=tv_start,
        tv_end_minute=tv_end,
        sunrise_minute=0,
        sunset_minute=0,
        cloudiness=0.0,
    )


def sample_away_blocks(
    rng: random.Random,
    persona: Persona,
    day_plan: DayPlan,
) -> list[tuple[int, int]]:
    if persona.name == "traveler" and day_plan.day_index % 12 in {9, 10, 11}:
        return [(0, 1440)]

    blocks: list[tuple[int, int]] = []

    def add_block(start: int, end: int) -> None:
        if start == end:
            return
        if end > start:
            blocks.append((start, end))
            return
        blocks.append((start, 1440))
        blocks.append((0, end))

    def safe_randint(low: int, high: int) -> int | None:
        if low > high:
            return None
        return rng.randint(low, high)

    if day_plan.leave_home_minute is not None and day_plan.return_home_minute is not None:
        add_block(day_plan.leave_home_minute, day_plan.return_home_minute)

    if persona.name == "gig_driver":
        block_count = rng.randint(2, 3)
        for _ in range(block_count):
            start = safe_randint(
                max(day_plan.wake_minute + 60, 360),
                max(day_plan.wake_minute + 120, 1080),
            )
            if start is None:
                continue
            duration = rng.randint(45, 150)
            add_block(start, min(1440, start + duration))
    elif persona.name == "student":
        if rng.random() < 0.65:
            start = safe_randint(max(day_plan.wake_minute + 60, 600), 1020)
            if start is not None:
                duration = rng.randint(60, 210)
                add_block(start, min(1440, start + duration))
        if rng.random() < 0.30:
            start = safe_randint(1140, 1320)
            if start is not None:
                duration = rng.randint(60, 150)
                add_block(start, min(1440, start + duration))
    elif persona.name in {"wfh", "hybrid", "retiree"} and rng.random() < persona.outing_probability:
        start = safe_randint(
            max(day_plan.wake_minute + 120, 600),
            max(day_plan.dinner_minute - 60, 720),
        )
        if start is not None:
            duration = rng.randint(*persona.outing_duration_minutes)
            add_block(start, min(1440, start + duration))
    elif persona.name == "social" and day_plan.day_of_week in {2, 4, 5}:
        start = safe_randint(max(day_plan.dinner_minute + 60, 1200), 1320)
        if start is not None:
            end = min(1440, start + rng.randint(90, 180))
            add_block(start, end)
    elif rng.random() < persona.outing_probability:
        start = safe_randint(
            max(day_plan.dinner_minute + 30, 1020),
            max(day_plan.wind_down_minute - 30, 1080),
        )
        if start is not None:
            duration = rng.randint(*persona.outing_duration_minutes)
            add_block(start, min(1440, start + duration))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(blocks):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


def sample_appliance_events(
    rng: random.Random,
    persona: Persona,
    day_index: int,
    day_plan: DayPlan | None = None,
) -> dict[str, list[tuple[int, int]]]:
    day_of_week = day_index % 7
    events: dict[str, list[tuple[int, int]]] = {
        "washing_machine_plug": [],
        "dryer_plug": [],
        "dishwasher_plug": [],
        "robot_vacuum_plug": [],
    }

    if day_of_week in persona.laundry_days:
        start = day_plan.dinner_minute + 30 if day_plan is not None else 1080
        washing_start = min(1380, start + rng.randint(0, 45))
        washing_end = min(1440, washing_start + 60)
        dryer_end = min(1440, washing_end + 60)
        events["washing_machine_plug"].append((washing_start, washing_end))
        events["dryer_plug"].append((washing_end, dryer_end))

    if rng.random() < persona.dishwasher_prob_per_day:
        start = (day_plan.dinner_minute if day_plan is not None else 1110) + 30
        events["dishwasher_plug"].append((min(1410, start), min(1440, start + 90)))

    if day_of_week in persona.robot_vacuum_days:
        start = max(540, day_plan.wake_minute + 120) if day_plan is not None else 540
        events["robot_vacuum_plug"].append((min(1200, start), min(1320, start + 60)))

    return events


def is_home(minute: int, away_blocks: list[tuple[int, int]]) -> bool:
    return not any(start <= minute < end for start, end in away_blocks)
