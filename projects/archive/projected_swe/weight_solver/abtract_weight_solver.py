import properscoring as ps
from typing import Dict, Tuple

from scipy.special import softmax
import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from projects.archive.projected_swe.weight_solver.indicator import AbstractIndicator, ReturnLevelComputationException, \
    ReturnLevel30YearsIndicator


class AbstractWeightSolver(object):

    def __init__(self, observation_study: AbstractStudy,
                 couple_to_historical_study: Dict[Tuple[str, str], AbstractStudy],
                 indicator_class: type,
                 massif_names=None,
                 add_interdependence_weight=False):
        self.observation_study = observation_study
        self.couple_to_historical_study = couple_to_historical_study
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

    # Prediction on the future period

    def weighted_projected_expected_indicator(self, massif_name, couple_to_projected_study):
        couple_to_projected_expected_indicator = self.couple_to_projected_expected_indicator(massif_name,
                                                                                             couple_to_projected_study)
        return sum([couple_to_projected_expected_indicator[c] * w for c, w in self.couple_to_weight.items()])

    def couple_to_projected_expected_indicator(self, massif_name, couple_to_projected_study):
        assert issubclass(self.indicator_class, AbstractIndicator)
        return {c: self.indicator_class.get_indicator(s, massif_name) for c, s in couple_to_projected_study.items()}

    def prediction_score(self, massif_name, couple_to_projected_study, projected_observation_study):
        try:
            target = self.target(massif_name, projected_observation_study)
            couple_to_projected_indicator = self.couple_to_projected_expected_indicator(massif_name,
                                                                                        couple_to_projected_study)
            couples, ensemble = zip(*list(couple_to_projected_indicator.items()))
            couple_to_weight = self.couple_to_weight
            weights = [couple_to_weight[c] for c in couples]
            crps_weighted = ps.crps_ensemble(target, ensemble, weights=weights)
            nb_weights = len(weights)
            weights_unweighted = [1 / nb_weights for _ in range(nb_weights)]
            crps_unweighted = ps.crps_ensemble(target, ensemble, weights=weights_unweighted)
            crpss = 100 * (crps_weighted - crps_unweighted) / crps_unweighted
            return crpss
        except ReturnLevelComputationException:
            return np.nan

    def mean_prediction_score(self, massif_names, couple_to_projected_study, projected_observation_study):
        scores = [self.prediction_score(massif_name, couple_to_projected_study, projected_observation_study) for
                  massif_name in massif_names]
        scores_filtered = [s for s in scores if not np.isnan(s)]
        assert len(scores_filtered) > 0
        return np.mean(scores_filtered)

    def target(self, massif_name, projected_observation_study):
        assert issubclass(self.indicator_class, AbstractIndicator)
        if self.indicator_class is ReturnLevel30YearsIndicator:
            return self.indicator_class.get_indicator(projected_observation_study, massif_name, bootstrap=True).mean()
        else:
            return self.indicator_class.get_indicator(projected_observation_study, massif_name)

    # Weight computation on the historical period

    @property
    def study_list(self):
        return [self.observation_study] + list(self.couple_to_historical_study.values())

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
                for couple, couple_study in self.couple_to_historical_study.items()}

    def compute_skill_nllh(self, couple_study):
        raise NotImplementedError

    @property
    def couple_to_nllh_interdependence(self):
        return {couple: self.compute_interdependence_nllh(couple_study=couple_study)
                for couple, couple_study in self.couple_to_historical_study.items()}

    def compute_interdependence_nllh(self, couple_study):
        raise NotImplementedError
