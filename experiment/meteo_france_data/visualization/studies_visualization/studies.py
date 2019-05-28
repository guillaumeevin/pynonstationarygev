from collections import OrderedDict
from typing import Dict

from experiment.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from experiment.meteo_france_data.scm_models_data.scm_constants import ALTITUDES


class Studies(object):
    """Object that will handle studies of the same study type (it could be Safran for instance)
    at several altitudes"""

    def __init__(self, study_type, altitude_list=None) -> None:
        # Load altitude_list attribute
        if altitude_list is None:
            altitude_list = ALTITUDES
        else:
            assert isinstance(altitude_list, list)
            assert len(altitude_list) > 0
            assert all([altitudes in ALTITUDES for altitudes in altitude_list])
            altitude_list = sorted(altitude_list)
        self.altitude_list = altitude_list
        # Load altitude_to_study attribute
        self.altitude_to_study = OrderedDict()  # type: Dict[int, AbstractStudy]
        for altitude in self.altitude_list:
            self.altitude_to_study[altitude] = study_type(altitude=altitude)

    @property
    def first_study(self):
        return self.altitude_to_study[self.altitude_list[0]]
