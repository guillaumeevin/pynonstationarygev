import numpy as np
from scipy.stats import norm

from projects.projected_swe.weight_solver.abtract_weight_solver import AbstractWeightSolver
from projects.projected_swe.weight_solver.indicator import AbstractIndicator, NllhComputationException, \
    WeightComputationException


class KnuttiWeightSolver(AbstractWeightSolver):

    def __init__(self, sigma_skill, sigma_interdependence, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_skill = sigma_skill
        self.sigma_interdependence = sigma_interdependence
        # Compute the subset of massif_names used for the computation
        self.massif_names_for_computation = []
        for massif_name in self.massif_names:
            try:
                [self.compute_skill_one_massif(couple_study, massif_name) for couple_study in self.study_list]
            except WeightComputationException:
                continue
            self.massif_names_for_computation.append(massif_name)
        assert len(self.massif_names_for_computation) > 0, 'Sigma values should be increased'

    @property
    def nb_massifs_for_computation(self):
        return len(self.massif_names)

    def compute_skill_nllh(self, couple_study):
        return sum([self.compute_skill_one_massif(couple_study, massif_name)
                    for massif_name in self.massif_names_for_computation])

    def compute_interdependence_nllh(self, couple_study):
        return sum([self.compute_interdependence_nllh_one_massif(couple_study, massif_name)
                    for massif_name in self.massif_names_for_computation])

    def compute_skill_one_massif(self, couple_study, massif_name):
        return self.compute_nllh_from_two_study(self.observation_study, couple_study, self.sigma_skill, massif_name)

    def compute_interdependence_nllh_one_massif(self, couple_study, massif_name):
        sum_proba = 0
        for other_couple_study in self.couple_to_study.values():
            if other_couple_study is not couple_study:
                nllh = self.compute_nllh_from_two_study(couple_study, other_couple_study,
                                                        self.sigma_interdependence, massif_name)
                proba = np.exp(-nllh)
                sum_proba += proba
        proba = 1 / (1 + sum_proba)
        nllh = -np.log(proba)
        return nllh

    def compute_nllh_from_two_study(self, study_1, study_2, sigma, massif_name):
        differences = self.sum_of_differences(study_1, study_2, massif_name)
        scale = np.sqrt(np.power(sigma, 2) * self.nb_massifs_for_computation / 2)
        proba = norm.pdf(differences, 0, scale)
        if not(0 < proba <= 1):
            raise NllhComputationException
        nllh = -np.log(proba)
        return nllh.sum()

    def sum_of_differences(self, study_1, study_2, massif_name):
        assert issubclass(self.indicator_class, AbstractIndicator)
        return np.array([self.indicator_class.get_indicator(study_1, massif_name)
                         - self.indicator_class.get_indicator(study_2, massif_name)])


class KnuttiWeightSolverWithBootstrapVersion1(KnuttiWeightSolver):

    def sum_of_differences(self, study_1, study_2, massif_name):
        assert issubclass(self.indicator_class, AbstractIndicator)
        bootstrap_study_1 = self.indicator_class.get_indicator(study_1, massif_name, bootstrap=True)
        bootstrap_study_2 = self.indicator_class.get_indicator(study_2, massif_name, bootstrap=True)
        differences = bootstrap_study_1 - bootstrap_study_2
        squared_difference = np.power(differences, 2)
        return squared_difference.sum()


class KnuttiWeightSolverWithBootstrapVersion2(KnuttiWeightSolver):

    def sum_of_differences(self, study_1, study_2, massif_name):
        assert issubclass(self.indicator_class, AbstractIndicator)
        bootstrap_study_1 = self.indicator_class.get_indicator(study_1, massif_name, bootstrap=True)
        bootstrap_study_2 = self.indicator_class.get_indicator(study_2, massif_name, bootstrap=True)
        differences = np.subtract.outer(bootstrap_study_1, bootstrap_study_2)
        squared_difference = np.power(differences, 2)
        return squared_difference.sum()
