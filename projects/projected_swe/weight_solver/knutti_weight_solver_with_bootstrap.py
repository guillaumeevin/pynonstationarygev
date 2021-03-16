import numpy as np

from projects.projected_swe.weight_solver.indicator import AbstractIndicator
from projects.projected_swe.weight_solver.knutti_weight_solver import KnuttiWeightSolver


class KnuttiWeightSolverWithBootstrap(KnuttiWeightSolver):

    def couple_to_projected_expected_indicator(self, massif_name, couple_to_projected_study):
        assert issubclass(self.indicator_class, AbstractIndicator)
        return {c: self.indicator_class.get_indicator(s, massif_name, bootstrap=True).mean()
                for c, s in couple_to_projected_study.items()}


class KnuttiWeightSolverWithBootstrapVersion1(KnuttiWeightSolverWithBootstrap):

    def differences(self, study_1, study_2, massif_name):
        assert issubclass(self.indicator_class, AbstractIndicator)
        bootstrap_study_1 = self.indicator_class.get_indicator(study_1, massif_name, bootstrap=True)
        bootstrap_study_2 = self.indicator_class.get_indicator(study_2, massif_name, bootstrap=True)
        differences = bootstrap_study_1 - bootstrap_study_2
        return differences


class KnuttiWeightSolverWithBootstrapVersion2(KnuttiWeightSolverWithBootstrap):

    def differences(self, study_1, study_2, massif_name):
        assert issubclass(self.indicator_class, AbstractIndicator)
        bootstrap_study_1 = self.indicator_class.get_indicator(study_1, massif_name, bootstrap=True)
        bootstrap_study_2 = self.indicator_class.get_indicator(study_2, massif_name, bootstrap=True)
        differences = np.subtract.outer(bootstrap_study_1, bootstrap_study_2)
        return differences
