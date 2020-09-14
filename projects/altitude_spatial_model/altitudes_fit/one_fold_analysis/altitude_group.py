from enum import Enum


class AltitudeGroup(Enum):
    low = 0
    mid = 1
    high = 2
    unspecfied = 3


def altitude_group_to_reference_altitude():
    return {
        AltitudeGroup.low: 1000,
        AltitudeGroup.mid: 2000,
        AltitudeGroup.high: 3000,
        AltitudeGroup.unspecfied: 1000,
    }

def get_altitude_group_from_altitudes(altitudes):
    s = set(altitudes)
    if s == {900, 1200, 1500}:
        return AltitudeGroup.low
    elif s == {1800, 2100, 2400}:
        return AltitudeGroup.mid
    elif s == {2700, 3000, 3300}:
        return AltitudeGroup.high
    else:
        return AltitudeGroup.unspecfied
