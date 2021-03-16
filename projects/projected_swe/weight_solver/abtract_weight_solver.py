from typing import Dict, Tuple

from scipy.special import softmax
import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from projects.projected_swe.weight_solver.indicator import AbstractIndicator


class AbstractWeightSolver(object):

    def __init__(self, observation_study: AbstractStudy,
                 couple_to_study: Dict[Tuple[str, str], AbstractStudy],
                 indicator_class: type,
                 massif_names=None,
                 add_interdependence_weight=False):
        self.observation_study = observation_study
        self.couple_to_study = couple_to_study
        self.indicator_class = indicator_class
        self.add_interdependence_weight = add_interdependence_weight
        # Compute intersection massif names
        sets = [set(study.study_massif_names) for study in self.study_list]
        intersection_massif_names = sets[0].intersection(*sets[1:])
        if massif_names is None:
            self.massif_names = list(intersection_massif_names)
        else:
            assert set(massif_names).issubset(intersection_massif_names)
            self.massif_names = massif_names

    @property
    def study_list(self):
        return [self.observation_study] + list(self.couple_to_study.values())

    @property
    def couple_to_weight(self):
        couple_list, nllh_list = zip(*list(self.couple_to_nllh.items()))
        weights = softmax(-np.array(nllh_list))
        return dict(zip(couple_list, weights))

    @property
    def couple_to_nllh(self):
        couple_to_nllh = self.couple_to_nllh_skill
        if self.add_interdependence_weight:
            for c, v in self.couple_to_nllh_interdependence.items():
                couple_to_nllh[c] += v
        return couple_to_nllh

    @property
    def couple_to_nllh_skill(self):
        return {couple: self.compute_skill_nllh(couple_study=couple_study)
                for couple, couple_study in self.couple_to_study.items()}

    def compute_skill_nllh(self, couple_study):
        raise NotImplementedError

    @property
    def couple_to_nllh_interdependence(self):
        return {couple: self.compute_interdependence_nllh(couple_study=couple_study)
                for couple, couple_study in self.couple_to_study.items()}

    def compute_interdependence_nllh(self, couple_study):
        raise NotImplementedError
