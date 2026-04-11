from __future__ import annotations

import math
import random


def outdoor_light_level(
    minute: int,
    sunrise: int,
    sunset: int,
    cloudiness: float,
    rng: random.Random | None = None,
) -> float:
    if minute <= sunrise or minute >= sunset:
        base_light = 0.0
    else:
        day_progress = (minute - sunrise) / max(1, sunset - sunrise)
        base_light = math.sin(math.pi * day_progress) ** 1.2

    noise = rng.gauss(0.0, 0.03) if rng is not None else 0.0
    noisy_light = base_light * cloudiness + noise
    return max(0.0, min(1.0, noisy_light))


def seasonal_sunrise_sunset(day_of_year: int) -> tuple[int, int]:
    angle = 2.0 * math.pi * (day_of_year - 172) / 365.0
    daylight_minutes = 720 + 180 * math.sin(angle)
    solar_noon = 12 * 60
    sunrise = round(solar_noon - daylight_minutes / 2.0)
    sunset = round(solar_noon + daylight_minutes / 2.0)
    return sunrise, sunset


def per_room_ambient(outdoor: float, curtain_factor: float) -> float:
    return max(0.0, min(1.0, outdoor * curtain_factor))


def indoor_temperature(minute: int, outdoor_temp: float, hvac_setpoint: float) -> float:
    circadian = math.sin((2.0 * math.pi * minute / 1440.0) - (math.pi / 2.0))
    drift = (outdoor_temp - hvac_setpoint) * 0.12
    return round(hvac_setpoint + drift + circadian * 0.35, 2)


def seasonal_outdoor_temperature(day_of_year: int, temp_range: tuple[float, float]) -> float:
    low, high = temp_range
    midpoint = (low + high) / 2.0
    amplitude = (high - low) / 2.0
    angle = 2.0 * math.pi * (day_of_year - 200) / 365.0
    return midpoint + amplitude * math.sin(angle)


def season_label(day_of_year: int) -> str:
    if 80 <= day_of_year < 172:
        return "spring"
    if 172 <= day_of_year < 264:
        return "summer"
    if 264 <= day_of_year < 355:
        return "fall"
    return "winter"
