from enum import Enum

# The order is important
altitudes_for_groups = [
    [300, 600, 900][1:],
    [1200, 1500, 1800],
    [2100, 2400, 2700],
    [3000, 3300, 3600, 3900]
]


class AbstractAltitudeGroup(object):

    @property
    def name(self):
        raise NotImplementedError

    @property
    def reference_altitude(self):
        raise NotImplementedError

    @property
    def xlabel(self):
        # warning: the label could not correspond to all massifs, some might have been fitted with less data
        # idx = get_index_group_from_reference_altitude(reference_altitude)
        # min_altitude, *_, max_altitude = altitudes_for_groups[idx]
        i = self.reference_altitude // 1000
        min_altitude, max_altitude = 1000 * i, 1000 * (i + 1)
        return 'Altitude = {} m\n' \
               'Estimated with maxima between {} m and {} m'.format(self.reference_altitude,
                                                                        min_altitude,
                                                                        max_altitude)


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
