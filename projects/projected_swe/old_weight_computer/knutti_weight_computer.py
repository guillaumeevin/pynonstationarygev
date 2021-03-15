import numpy as np

from projects.projected_swe.weight_computer.abstract_weight_computer import AbstractWeightComputer
from projects.projected_swe.weight_computer.non_stationary_weight_computer import NllhWeightComputer


class KnuttiWeightComputer(AbstractWeightComputer):

    def __init__(self, *args, sigma_D, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_D = sigma_D

    def compute_nllh_local(self, ensemble_fit, gcm_rcm_couple, massif_name, reference_study):
        mean_maxima_reference = reference_study.massif_name_to_annual_maxima[massif_name].mean()
        studies = ensemble_fit.gcm_rcm_couple_to_visualizer[gcm_rcm_couple].studies
        study = studies.altitude_to_study[reference_study.altitude]
        mean_maxima_couple = study.massif_name_to_annual_maxima[massif_name].mean()
        ratio = (mean_maxima_couple - mean_maxima_reference) / self.sigma_D
        proba = np.exp(-np.power(ratio, 2))
        return -np.log(proba)



class MixedWeightComputer(NllhWeightComputer, KnuttiWeightComputer):
    pass