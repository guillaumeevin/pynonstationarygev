import numpy as np
from projects.projected_swe.weight_solver.abtract_weight_solver import AbstractWeightSolver
from projects.projected_swe.weight_solver.indicator import AbstractIndicator


class KnuttiWeightSolver(AbstractWeightSolver):

    def __init__(self, sigma_skill, sigma_interdependence, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_skill = sigma_skill
        self.sigma_interdependence = sigma_interdependence

    def compute_skill(self, couple_study):
        raise self.compute_distance_between_two_study(self.observation_study, self.couple_to_study, self.sigma_skill)

    def compute_interdependence(self, couple_study):
        sum = 0
        for other_couple_study in self.couple_to_study.values():
            if other_couple_study is not couple_study:
                sum += self.compute_distance_between_two_study(couple_study, other_couple_study, self.sigma_interdependence)
        return 1 / (1 + sum)

    def compute_distance_between_two_study(self, study_1, study_2, sigma):
        difference = self.sum_of_differences(study_1, study_2)
        return np.exp(-np.power(difference, 2 * sigma))

    def sum_of_differences(self, study_1, study_2):
        assert issubclass(self.indicator_class, AbstractIndicator)
        return self.indicator_class.get_indicator(study_1) - self.indicator_class.get_indicator(study_2)


class KnuttiWeightSolverWithBootstrapVersion1(KnuttiWeightSolver):

    def sum_of_differences(self, study_1, study_2):
        assert issubclass(self.indicator_class, AbstractIndicator)
        bootstrap_study_1 = self.indicator_class.get_indicator(study_1, bootstrap=True)
        bootstrap_study_2 = self.indicator_class.get_indicator(study_2, bootstrap=True)
        differences = bootstrap_study_1 - bootstrap_study_2
        return differences.sum()


class KnuttiWeightSolverWithBootstrapVersion2(KnuttiWeightSolver):

    def sum_of_differences(self, study_1, study_2):
        assert issubclass(self.indicator_class, AbstractIndicator)
        bootstrap_study_1 = self.indicator_class.get_indicator(study_1, bootstrap=True)
        bootstrap_study_2 = self.indicator_class.get_indicator(study_2, bootstrap=True)
        differences = np.subtract.outer(bootstrap_study_1, bootstrap_study_2)
        return differences.sum()
