from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeDist:
    mean: int
    stddev: int
    min: int
    max: int


@dataclass(frozen=True)
class Persona:
    name: str
    wake: TimeDist
    leave_home: TimeDist | None
    return_home: TimeDist | None
    wind_down: TimeDist
    sleep: TimeDist
    breakfast: TimeDist | None
    lunch: TimeDist | None
    dinner: TimeDist
    tv_start: TimeDist | None
    tv_duration_minutes: tuple[int, int]
    laundry_days: tuple[int, ...]
    dishwasher_prob_per_day: float
    robot_vacuum_days: tuple[int, ...]
    outing_probability: float
    outing_duration_minutes: tuple[int, int]
    weekend_shift_minutes: int
    primary_daytime_room: str
    bedroom_curtain_factor: float = 0.35
    hvac_setpoint_c: float = 21.0
    outdoor_temp_c: tuple[float, float] = (-2.0, 24.0)


PERSONAS: dict[str, Persona] = {
    "commuter": Persona(
        name="commuter",
        wake=TimeDist(390, 20, 360, 450),
        leave_home=TimeDist(450, 15, 420, 495),
        return_home=TimeDist(1050, 30, 960, 1110),
        wind_down=TimeDist(1320, 15, 1290, 1350),
        sleep=TimeDist(1380, 15, 1350, 1410),
        breakfast=TimeDist(415, 15, 390, 465),
        lunch=None,
        dinner=TimeDist(1125, 20, 1080, 1170),
        tv_start=TimeDist(1140, 25, 1080, 1230),
        tv_duration_minutes=(60, 150),
        laundry_days=(2, 6),
        dishwasher_prob_per_day=0.65,
        robot_vacuum_days=(5,),
        outing_probability=0.15,
        outing_duration_minutes=(45, 120),
        weekend_shift_minutes=90,
        primary_daytime_room="none",
    ),
    "wfh": Persona(
        name="wfh",
        wake=TimeDist(465, 25, 420, 540),
        leave_home=None,
        return_home=None,
        wind_down=TimeDist(1335, 20, 1290, 1380),
        sleep=TimeDist(1395, 15, 1350, 1425),
        breakfast=TimeDist(495, 20, 450, 555),
        lunch=TimeDist(735, 20, 690, 795),
        dinner=TimeDist(1110, 20, 1050, 1170),
        tv_start=TimeDist(1170, 30, 1110, 1260),
        tv_duration_minutes=(45, 120),
        laundry_days=(1, 4),
        dishwasher_prob_per_day=0.75,
        robot_vacuum_days=(2, 5),
        outing_probability=0.25,
        outing_duration_minutes=(30, 75),
        weekend_shift_minutes=60,
        primary_daytime_room="office",
    ),
    "hybrid": Persona(
        name="hybrid",
        wake=TimeDist(435, 25, 390, 510),
        leave_home=TimeDist(465, 20, 435, 525),
        return_home=TimeDist(1035, 35, 945, 1110),
        wind_down=TimeDist(1320, 20, 1275, 1365),
        sleep=TimeDist(1380, 20, 1335, 1410),
        breakfast=TimeDist(465, 20, 420, 540),
        lunch=TimeDist(735, 25, 690, 810),
        dinner=TimeDist(1110, 20, 1050, 1170),
        tv_start=TimeDist(1170, 30, 1110, 1260),
        tv_duration_minutes=(45, 120),
        laundry_days=(3, 6),
        dishwasher_prob_per_day=0.70,
        robot_vacuum_days=(1, 5),
        outing_probability=0.20,
        outing_duration_minutes=(30, 90),
        weekend_shift_minutes=75,
        primary_daytime_room="office",
    ),
    "night_shift": Persona(
        name="night_shift",
        wake=TimeDist(960, 25, 900, 1020),
        leave_home=TimeDist(1260, 20, 1230, 1320),
        return_home=TimeDist(450, 20, 420, 510),
        wind_down=TimeDist(510, 20, 465, 555),
        sleep=TimeDist(540, 20, 495, 585),
        breakfast=TimeDist(990, 20, 945, 1050),
        lunch=TimeDist(30, 15, 0, 90),
        dinner=TimeDist(1140, 20, 1080, 1200),
        tv_start=TimeDist(870, 30, 780, 960),
        tv_duration_minutes=(60, 150),
        laundry_days=(2, 5),
        dishwasher_prob_per_day=0.55,
        robot_vacuum_days=(3,),
        outing_probability=0.08,
        outing_duration_minutes=(30, 60),
        weekend_shift_minutes=45,
        primary_daytime_room="none",
        bedroom_curtain_factor=0.05,
        outdoor_temp_c=(-4.0, 22.0),
    ),
    "early_shift": Persona(
        name="early_shift",
        wake=TimeDist(285, 15, 255, 315),
        leave_home=TimeDist(330, 15, 300, 360),
        return_home=TimeDist(930, 20, 870, 990),
        wind_down=TimeDist(1200, 15, 1170, 1230),
        sleep=TimeDist(1260, 15, 1230, 1290),
        breakfast=TimeDist(300, 15, 270, 345),
        lunch=None,
        dinner=TimeDist(1020, 20, 975, 1080),
        tv_start=TimeDist(1050, 20, 1005, 1110),
        tv_duration_minutes=(30, 90),
        laundry_days=(4,),
        dishwasher_prob_per_day=0.55,
        robot_vacuum_days=(6,),
        outing_probability=0.10,
        outing_duration_minutes=(30, 75),
        weekend_shift_minutes=60,
        primary_daytime_room="none",
        outdoor_temp_c=(-6.0, 20.0),
    ),
    "retiree": Persona(
        name="retiree",
        wake=TimeDist(420, 20, 390, 480),
        leave_home=None,
        return_home=None,
        wind_down=TimeDist(1260, 15, 1230, 1290),
        sleep=TimeDist(1320, 15, 1290, 1350),
        breakfast=TimeDist(465, 20, 420, 525),
        lunch=TimeDist(720, 20, 675, 780),
        dinner=TimeDist(1020, 20, 975, 1080),
        tv_start=TimeDist(840, 30, 780, 930),
        tv_duration_minutes=(90, 180),
        laundry_days=(1, 5),
        dishwasher_prob_per_day=0.60,
        robot_vacuum_days=(2,),
        outing_probability=0.30,
        outing_duration_minutes=(45, 120),
        weekend_shift_minutes=30,
        primary_daytime_room="living_room",
    ),
    "student": Persona(
        name="student",
        wake=TimeDist(570, 45, 480, 690),
        leave_home=None,
        return_home=None,
        wind_down=TimeDist(30, 30, 0, 90),
        sleep=TimeDist(60, 20, 30, 120),
        breakfast=TimeDist(615, 25, 540, 720),
        lunch=TimeDist(810, 30, 720, 900),
        dinner=TimeDist(1170, 30, 1080, 1260),
        tv_start=TimeDist(1320, 40, 1230, 1410),
        tv_duration_minutes=(60, 180),
        laundry_days=(6,),
        dishwasher_prob_per_day=0.40,
        robot_vacuum_days=(5,),
        outing_probability=0.50,
        outing_duration_minutes=(60, 180),
        weekend_shift_minutes=90,
        primary_daytime_room="bedroom",
    ),
    "gig_driver": Persona(
        name="gig_driver",
        wake=TimeDist(540, 30, 480, 615),
        leave_home=None,
        return_home=None,
        wind_down=TimeDist(1410, 20, 1350, 1435),
        sleep=TimeDist(30, 20, 0, 75),
        breakfast=TimeDist(570, 20, 510, 645),
        lunch=TimeDist(780, 25, 720, 840),
        dinner=TimeDist(1170, 25, 1110, 1230),
        tv_start=TimeDist(1260, 35, 1170, 1350),
        tv_duration_minutes=(30, 90),
        laundry_days=(2,),
        dishwasher_prob_per_day=0.35,
        robot_vacuum_days=(1,),
        outing_probability=0.80,
        outing_duration_minutes=(45, 120),
        weekend_shift_minutes=45,
        primary_daytime_room="none",
    ),
    "traveler": Persona(
        name="traveler",
        wake=TimeDist(390, 20, 360, 450),
        leave_home=TimeDist(450, 15, 420, 495),
        return_home=TimeDist(1050, 30, 960, 1110),
        wind_down=TimeDist(1320, 15, 1290, 1350),
        sleep=TimeDist(1380, 15, 1350, 1410),
        breakfast=TimeDist(420, 20, 390, 480),
        lunch=None,
        dinner=TimeDist(1125, 20, 1080, 1170),
        tv_start=TimeDist(1170, 30, 1110, 1260),
        tv_duration_minutes=(45, 120),
        laundry_days=(6,),
        dishwasher_prob_per_day=0.55,
        robot_vacuum_days=(1,),
        outing_probability=0.10,
        outing_duration_minutes=(45, 120),
        weekend_shift_minutes=75,
        primary_daytime_room="none",
    ),
    "social": Persona(
        name="social",
        wake=TimeDist(405, 25, 360, 480),
        leave_home=TimeDist(465, 15, 435, 510),
        return_home=TimeDist(1050, 30, 990, 1110),
        wind_down=TimeDist(15, 20, 0, 75),
        sleep=TimeDist(45, 20, 15, 105),
        breakfast=TimeDist(435, 20, 390, 510),
        lunch=None,
        dinner=TimeDist(1140, 20, 1080, 1200),
        tv_start=TimeDist(1245, 30, 1170, 1320),
        tv_duration_minutes=(30, 90),
        laundry_days=(6,),
        dishwasher_prob_per_day=0.50,
        robot_vacuum_days=(3,),
        outing_probability=0.45,
        outing_duration_minutes=(90, 180),
        weekend_shift_minutes=105,
        primary_daytime_room="none",
    ),
}
