from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta

from fabricator.ambient import (
    indoor_temperature,
    outdoor_light_level,
    season_label,
    seasonal_outdoor_temperature,
    seasonal_sunrise_sunset,
)
from fabricator.personas import PERSONAS, Persona
from fabricator.rules import (
    AnomalyConfig,
    apply_anomalies,
    compute_ambient,
    compute_lights,
    compute_plugs,
    compute_presence,
)
from fabricator.schedule import (
    sample_appliance_events,
    sample_away_blocks,
    sample_day_schedule,
    is_home,
)


@dataclass(frozen=True)
class EventRow:
    step_index: int
    week_index: int
    day_index: int
    timestamp: str
    persona_label: str
    season_label: str
    sunrise_minute: int
    sunset_minute: int
    cloudiness: float
    time_of_day_norm: float
    day_of_week: int
    is_weekend: bool
    kitchen_light: float
    bedroom_light: float
    living_room_light: float
    bathroom_light: float
    hallway_light: float
    office_light: float
    dining_room_light: float
    bedside_lamp: float
    porch_light: float
    tv_living_room: float
    tv_bedroom: float
    coffee_maker_plug: float
    kettle_plug: float
    microwave_plug: float
    desk_monitor_plug: float
    game_console_plug: float
    washing_machine_plug: float
    dryer_plug: float
    dishwasher_plug: float
    robot_vacuum_plug: float
    space_heater_plug: float
    person_home: float
    front_door: float
    motion_kitchen: float
    motion_bedroom: float
    motion_living_room: float
    motion_bathroom: float
    motion_hallway: float
    motion_office: float
    ambient_light_outdoor: float
    ambient_light_bedroom: float
    indoor_temperature_c: float


class Fabricator:
    def __init__(
        self,
        persona: Persona | str,
        seed: int,
        days: int = 7,
        step_minutes: int = 15,
        anomaly_config: AnomalyConfig | None = None,
    ) -> None:
        self.persona = PERSONAS[persona] if isinstance(persona, str) else persona
        self.seed = seed
        self.days = days
        self.step_minutes = step_minutes
        self.anomaly_config = anomaly_config or AnomalyConfig()
        self.base_date = date(2024, 1, 1) + timedelta(days=seed % 365)

    def generate(self) -> list[EventRow]:
        return self._generate_rows(num_weeks=1)

    def generate_dataset(self, num_weeks: int) -> list[EventRow]:
        return self._generate_rows(num_weeks=num_weeks)

    def _generate_rows(self, num_weeks: int) -> list[EventRow]:
        rows: list[EventRow] = []
        for week_index in range(num_weeks):
            rng = random.Random(self.seed + week_index)
            for local_day_index in range(self.days):
                absolute_day_index = week_index * self.days + local_day_index
                day_date = self.base_date + timedelta(days=absolute_day_index)
                day_of_year = day_date.timetuple().tm_yday
                plan = sample_day_schedule(rng, self.persona, absolute_day_index)
                sunrise, sunset = seasonal_sunrise_sunset(day_of_year)
                cloudiness = rng.uniform(0.78, 1.0)
                plan = plan.__class__(**{**plan.__dict__, "sunrise_minute": sunrise, "sunset_minute": sunset, "cloudiness": cloudiness})
                away_blocks = sample_away_blocks(rng, self.persona, plan)
                appliance_events = sample_appliance_events(rng, self.persona, absolute_day_index, day_plan=plan)
                outdoor_temp = seasonal_outdoor_temperature(day_of_year, self.persona.outdoor_temp_c)
                label = season_label(day_of_year)

                for step_in_day in range(1440 // self.step_minutes):
                    minute = step_in_day * self.step_minutes
                    home_now = is_home(minute, away_blocks)
                    outdoor_light = outdoor_light_level(
                        minute,
                        sunrise=plan.sunrise_minute,
                        sunset=plan.sunset_minute,
                        cloudiness=plan.cloudiness,
                        rng=rng,
                    )
                    lights = compute_lights(self.persona, plan, minute, home_now, outdoor_light)
                    plugs = compute_plugs(
                        self.persona,
                        plan,
                        minute,
                        home_now,
                        appliance_events,
                        lights=lights,
                        outdoor_temp=outdoor_temp,
                    )
                    presence = compute_presence(
                        self.persona,
                        plan,
                        minute,
                        home_now,
                        away_blocks,
                        lights=lights,
                        rng=rng,
                    )
                    ambient = compute_ambient(
                        plan,
                        minute,
                        outdoor_light,
                        indoor_temperature(minute, outdoor_temp, self.persona.hvac_setpoint_c),
                        self.persona.bedroom_curtain_factor,
                    )

                    timestamp = datetime.combine(day_date, time(minute // 60, minute % 60)).isoformat()
                    snapshot: dict[str, float | int | str | bool] = {
                        "step_index": len(rows),
                        "week_index": week_index,
                        "day_index": absolute_day_index,
                        "timestamp": timestamp,
                        "persona_label": self.persona.name,
                        "season_label": label,
                        "sunrise_minute": plan.sunrise_minute,
                        "sunset_minute": plan.sunset_minute,
                        "cloudiness": round(plan.cloudiness, 4),
                        "time_of_day_norm": ambient["time_of_day_norm"],
                        "day_of_week": int(plan.day_of_week),
                        "is_weekend": bool(plan.is_weekend),
                        **lights,
                        **plugs,
                        **presence,
                        "ambient_light_outdoor": ambient["ambient_light_outdoor"],
                        "ambient_light_bedroom": ambient["ambient_light_bedroom"],
                        "indoor_temperature_c": ambient["indoor_temperature_c"],
                    }
                    apply_anomalies(
                        snapshot,
                        self.persona,
                        plan,
                        minute,
                        home_now,
                        rng,
                        self.anomaly_config,
                    )
                    rows.append(EventRow(**snapshot))
        return rows

    @staticmethod
    def as_dicts(rows: list[EventRow]) -> list[dict[str, object]]:
        return [asdict(row) for row in rows]
