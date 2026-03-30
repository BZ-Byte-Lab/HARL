from __future__ import annotations

import math
import random
from dataclasses import dataclass

from config import PatternConfig


def action_to_device_vector(action: int, num_devices: int = 5) -> list[int]:
    return [(action >> bit_index) & 1 for bit_index in range(num_devices)]


def device_vector_to_action(device_vector: list[int]) -> int:
    action = 0
    for bit_index, value in enumerate(device_vector):
        action |= (int(value) & 1) << bit_index
    return action


@dataclass(frozen=True)
class DailyPlan:
    is_weekend: bool
    wake_minute: int
    leave_home_minute: int | None
    return_home_minute: int | None
    wind_down_minute: int
    sleep_minute: int
    outing_start_minute: int | None
    outing_end_minute: int | None
    tv_start_minute: int | None
    tv_end_minute: int | None
    sunrise_minute: int
    sunset_minute: int
    cloudiness: float


@dataclass(frozen=True)
class WorldStep:
    step_index: int
    day_index: int
    step_in_day: int
    time_of_day_norm: float
    weekday_norm: float
    is_home: int
    light_level: float
    ideal_devices: list[int]


class WeekPatternWorld:
    def __init__(self, config: PatternConfig, seed: int | None = None) -> None:
        self.config = config
        self.seed = config.seed if seed is None else seed
        self.rng = random.Random(self.seed)

    def generate_episode(self) -> list[WorldStep]:
        steps: list[WorldStep] = []

        for day_index in range(self.config.episode_days):
            plan = self._generate_day_plan(day_index)
            for step_in_day in range(self.config.steps_per_day):
                minute = step_in_day * self.config.step_minutes
                light_level = self._light_level(plan, minute)
                is_home = int(self._is_home(plan, minute))
                ideal_devices = self._ideal_devices(plan, minute, light_level, is_home)
                steps.append(
                    WorldStep(
                        step_index=len(steps),
                        day_index=day_index,
                        step_in_day=step_in_day,
                        time_of_day_norm=minute / 1440.0,
                        weekday_norm=day_index / max(1, self.config.episode_days - 1),
                        is_home=is_home,
                        light_level=light_level,
                        ideal_devices=ideal_devices,
                    )
                )

        return steps

    def _generate_day_plan(self, day_index: int) -> DailyPlan:
        is_weekend = day_index >= 5

        if is_weekend:
            wake_minute = self._jittered_minute(8 * 60 + 30, 20, 8 * 60, 9 * 60)
            sleep_minute = self._jittered_minute(23 * 60 + 30, 20, 23 * 60, 23 * 60 + 55)
            wind_down_minute = min(
                sleep_minute - 30,
                self._jittered_minute(22 * 60 + 30, 15, 22 * 60, 23 * 60),
            )
            leave_home_minute = None
            return_home_minute = None
            outing_start_minute, outing_end_minute = self._weekend_outing()
            tv_start_minute = self._jittered_minute(15 * 60 + 30, 35, 14 * 60, 17 * 60)
            tv_end_minute = min(tv_start_minute + self.rng.randint(60, 180), wind_down_minute)
        else:
            wake_minute = self._jittered_minute(6 * 60 + 30, 20, 6 * 60, 7 * 60 + 30)
            leave_home_minute = self._jittered_minute(7 * 60 + 30, 15, 7 * 60, 8 * 60 + 15)
            return_home_minute = self._jittered_minute(17 * 60, 30, 16 * 60, 18 * 60 + 30)
            wind_down_minute = self._jittered_minute(22 * 60, 15, 21 * 60 + 30, 22 * 60 + 30)
            sleep_minute = self._jittered_minute(23 * 60, 15, 22 * 60 + 30, 23 * 60 + 30)
            outing_start_minute, outing_end_minute = self._weekday_outing(wind_down_minute)
            tv_start_minute = max(return_home_minute + 45, 18 * 60)
            tv_end_minute = wind_down_minute

        sunrise_minute = self._jittered_minute(6 * 60 + 30, 10, 6 * 60, 7 * 60)
        sunset_minute = self._jittered_minute(18 * 60, 20, 17 * 60, 18 * 60 + 30)
        cloudiness = self.rng.uniform(0.78, 1.00)

        return DailyPlan(
            is_weekend=is_weekend,
            wake_minute=wake_minute,
            leave_home_minute=leave_home_minute,
            return_home_minute=return_home_minute,
            wind_down_minute=wind_down_minute,
            sleep_minute=sleep_minute,
            outing_start_minute=outing_start_minute,
            outing_end_minute=outing_end_minute,
            tv_start_minute=tv_start_minute,
            tv_end_minute=tv_end_minute,
            sunrise_minute=sunrise_minute,
            sunset_minute=sunset_minute,
            cloudiness=cloudiness,
        )

    def _weekday_outing(self, wind_down_minute: int) -> tuple[int | None, int | None]:
        if self.rng.random() >= 0.12:
            return None, None

        start_minute = self._jittered_minute(20 * 60, 20, 19 * 60, 21 * 60)
        duration_minutes = self.rng.randint(30, 90)
        end_minute = min(start_minute + duration_minutes, wind_down_minute - 15)
        if end_minute <= start_minute:
            return None, None
        return start_minute, end_minute

    def _weekend_outing(self) -> tuple[int | None, int | None]:
        if self.rng.random() >= 0.15:
            return None, None

        start_minute = self._jittered_minute(14 * 60, 30, 12 * 60, 16 * 60)
        duration_minutes = self.rng.randint(60, 180)
        return start_minute, min(start_minute + duration_minutes, 20 * 60)

    def _is_home(self, plan: DailyPlan, minute: int) -> bool:
        if not plan.is_weekend:
            assert plan.leave_home_minute is not None
            assert plan.return_home_minute is not None
            if plan.leave_home_minute <= minute < plan.return_home_minute:
                return False

        if (
            plan.outing_start_minute is not None
            and plan.outing_end_minute is not None
            and plan.outing_start_minute <= minute < plan.outing_end_minute
        ):
            return False

        return True

    def _ideal_devices(
        self,
        plan: DailyPlan,
        minute: int,
        light_level: float,
        is_home: int,
    ) -> list[int]:
        devices = [0] * self.config.num_devices

        if not is_home or minute < plan.wake_minute or minute >= plan.sleep_minute:
            return devices

        morning_end = min(
            plan.wake_minute + 60,
            plan.leave_home_minute if plan.leave_home_minute is not None else plan.wake_minute + 90,
        )

        if plan.wake_minute <= minute < plan.wake_minute + 30:
            devices[1] = 1

        if plan.wake_minute <= minute < morning_end:
            devices[2] = 1

        if plan.wake_minute <= minute < plan.wake_minute + 45:
            devices[4] = 1

        if plan.is_weekend:
            if plan.tv_start_minute is not None and plan.tv_end_minute is not None:
                if plan.tv_start_minute <= minute < plan.tv_end_minute:
                    devices[3] = 1

            if plan.wind_down_minute <= minute < plan.sleep_minute:
                return [0, 1, 0, 0, 0]

            if light_level < 0.55:
                devices[0] = 1

            if 10 * 60 <= minute < 12 * 60 and light_level < 0.60:
                devices[4] = 1
        else:
            assert plan.return_home_minute is not None
            if plan.return_home_minute <= minute < min(plan.return_home_minute + 60, 19 * 60):
                devices[2] = 1

            if 18 * 60 <= minute < plan.wind_down_minute:
                devices[0] = 1
                devices[3] = 1
                if minute < 19 * 60:
                    devices[2] = 1

            if plan.wind_down_minute <= minute < plan.sleep_minute:
                return [0, 1, 0, 0, 0]

        if light_level < 0.25 and not any(devices[:3]):
            devices[0] = 1

        if light_level > 0.85:
            devices[0] = 0
            devices[1] = 0
            devices[2] = 0

        return devices

    def _light_level(self, plan: DailyPlan, minute: int) -> float:
        if minute <= plan.sunrise_minute or minute >= plan.sunset_minute:
            base_light = 0.0
        else:
            day_progress = (minute - plan.sunrise_minute) / (plan.sunset_minute - plan.sunrise_minute)
            base_light = math.sin(math.pi * day_progress) ** 1.2

        noisy_light = base_light * plan.cloudiness + self.rng.gauss(0.0, 0.03)
        return max(0.0, min(1.0, noisy_light))

    def _jittered_minute(self, base: int, stddev: int, minimum: int, maximum: int) -> int:
        jittered = round(self.rng.gauss(base, stddev))
        return max(minimum, min(maximum, jittered))

