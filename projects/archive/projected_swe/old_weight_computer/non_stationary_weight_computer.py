import numpy as np

from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import compute_nllh
from projects.archive.projected_swe import AbstractWeightComputer
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate


class NllhWeightComputer(AbstractWeightComputer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.visualizer.temporal_covariate_for_fit is TimeTemporalCovariate, \
            'for temperature covariate you should transform the coordinates'

    def compute_nllh_local(self, ensemble_fit, gcm_rcm_couple, massif_name, reference_study):
        coordinates = [np.array([reference_study.altitude, year]) for year in reference_study.ordered_years]
        maxima = reference_study.massif_name_to_annual_maxima[massif_name]
        best_function_from_fit = self.get_function_from_fit(ensemble_fit, massif_name, gcm_rcm_couple)
        # It is normal that it could crash, because some models where fitted with data smaller than
        # the data used to compute the nllh, therefore we use assertion_for_inf=False below
        return compute_nllh(coordinates, maxima, best_function_from_fit,
                            maximum_from_obs=False, assertion_for_inf=False)

    def get_function_from_fit(self, ensemble_fit, massif_name, gcm_rcm_couple):
        visualizer = ensemble_fit.gcm_rcm_couple_to_visualizer[gcm_rcm_couple]
        one_fold_fit = visualizer.massif_name_to_one_fold_fit[massif_name]
        return one_fold_fit.best_function_from_fit
