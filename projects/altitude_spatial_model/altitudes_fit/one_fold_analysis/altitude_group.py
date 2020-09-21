from enum import Enum

# The order is important
altitudes_for_groups = [
    [300, 600, 900],
    [1200, 1500, 1800],
    [2100, 2400, 2700],
    [3000, 3300, 3600, 3900]
]


# altitudes_for_groups = [
#     [900, 1200, 1500],
#     [1800, 2100, 2400],
#     [2700, 3000, 3300]
# ]

# altitudes_for_groups = [
#     [600, 900, 1200, 1500, 1800],
#     [1500, 1800, 2100, 2400, 2700],
#     [2400, 2700, 3000, 3300, 3600]
# ]


class AbstractAltitudeGroup(object):

    @property
    def name(self):
        raise NotImplementedError

    @property
    def reference_altitude(self):
        raise NotImplementedError


class LowAltitudeGroup(AbstractAltitudeGroup):

    @property
    def name(self):
        return 'low'

    @property
    def reference_altitude(self):
        return 600


class MidAltitudeGroup(AbstractAltitudeGroup):

    @property
    def name(self):
        return 'mid'

    @property
    def reference_altitude(self):
        return 1500


class HighAltitudeGroup(AbstractAltitudeGroup):

    @property
    def name(self):
        return 'high'

    @property
    def reference_altitude(self):
        return 2400


class VeyHighAltitudeGroup(AbstractAltitudeGroup):

    @property
    def name(self):
        return 'very high'

    @property
    def reference_altitude(self):
        return 3300


class DefaultAltitudeGroup(AbstractAltitudeGroup):

    @property
    def name(self):
        return 'default'

    @property
    def reference_altitude(self):
        return 500


def get_altitude_group_from_altitudes(altitudes):
    s = set(altitudes)
    if s == set(altitudes_for_groups[0]):
        return LowAltitudeGroup()
    elif s == set(altitudes_for_groups[1]):
        return MidAltitudeGroup()
    elif s == set(altitudes_for_groups[2]):
        return HighAltitudeGroup()
    elif s == set(altitudes_for_groups[3]):
        return VeyHighAltitudeGroup()
    else:
        return DefaultAltitudeGroup()
