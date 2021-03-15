from scipy.special import softmax
import numpy as np

from projects.projected_swe.weight_solver.indicator import AbstractIndicator


class AbstractWeightSolver(object):

    def __init__(self, observation_study, couple_to_study, indicator_class: type, add_interdependence_weight=False):
        self.observation_study = observation_study
        self.couple_to_study = couple_to_study
        self.indicator_class = indicator_class
        self.add_interdependence_weight = add_interdependence_weight

    @property
    def couple_to_weight(self):
        nllh_list, couple_list = zip(*list(self.couple_to_nllh.items()))
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
        couple_to_nllh_skill = {}
        for couple, couple_study in self.couple_to_study.items():
            skill = self.compute_skill(couple_study=couple_study)
            nllh_skill = -np.log(skill)
            couple_to_nllh_skill[couple] = nllh_skill
        return couple_to_nllh_skill

    def compute_skill(self, couple_study):
        raise NotImplementedError

    @property
    def couple_to_nllh_interdependence(self):
        couple_to_nllh_interdependence = {}
        for couple, couple_study in self.couple_to_study.items():
            interdependence = self.compute_interdependence(couple_study=couple_study)
            nllh_interdependence = -np.log(interdependence)
            couple_to_nllh_interdependence[couple] = nllh_interdependence
        return couple_to_nllh_interdependence

    def compute_interdependence(self, couple_study):
        raise NotImplementedError
