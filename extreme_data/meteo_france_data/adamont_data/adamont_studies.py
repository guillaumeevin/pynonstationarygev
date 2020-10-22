from collections import OrderedDict

from cached_property import cached_property

from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_full_name
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy


class AdamontStudies(object):

    def __init__(self, study_class, gcm_rcm_couples=None, **kwargs_study):
        self.study_class = study_class
        if gcm_rcm_couples is None:
            gcm_rcm_couples = list(gcm_rcm_couple_to_full_name.keys())
        self.gcm_rcm_couples = gcm_rcm_couples
        self.gcm_rcm_couples_to_study = OrderedDict()  # type: OrderedDict[int, AbstractStudy]
        for gcm_rcm_couple in self.gcm_rcm_couples:
            study = study_class(gcm_rcm_couple=gcm_rcm_couple, **kwargs_study)
            self.gcm_rcm_couples_to_study[gcm_rcm_couple] = study

    @property
    def study_list(self):
        return list(self.gcm_rcm_couples_to_study.values())

    @cached_property
    def study(self) -> AbstractStudy:
        return self.study_list[0]