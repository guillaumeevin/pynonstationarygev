import numpy as np

from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import compute_nllh
from projects.projected_swe.abstract_weight_computer import AbstractWeightComputer
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate


class NllhWeightComputer(AbstractWeightComputer):

    def compute_gcm_rcm_couple_to_local_nllh(self, ensemble_fit, gcm_rcm_couples_subset, massif_name, maxima,
                                             altitude, reference_study):
        assert ensemble_fit.temporal_covariate_for_fit is TimeTemporalCovariate, \
            'for temperature covariate you should transform the coordinates'
        gcm_rcm_couple_to_local_nllh = {}
        # Check that all the gcm_rcm_couple have a model for this massif_name
        if self.condition_to_compute_nllh(ensemble_fit, massif_name, self.visualizer):
            print(ensemble_fit.altitudes, massif_name)
            coordinates = [np.array([altitude, year]) for year in reference_study.ordered_years]
            nllh_list = []
            for gcm_rcm_couple in gcm_rcm_couples_subset:
                best_function_from_fit = self.get_function_from_fit(ensemble_fit, massif_name, gcm_rcm_couple)
                # It is normal that it could crash, because some models where fitted with data smaller than
                # the data used to compute the nllh

                nllh = compute_nllh(coordinates, maxima, best_function_from_fit,
                                    maximum_from_obs=False, assertion_for_inf=False)
                nllh_list.append(nllh)

            if all([not np.isinf(nllh) for nllh in nllh_list]):
                return dict(zip(gcm_rcm_couples_subset, nllh_list))

    def condition_to_compute_nllh(self, ensemble_fit, massif_name, visualizer):
        return all(
            [massif_name in ensemble_fit.gcm_rcm_couple_to_visualizer[c].massif_name_to_one_fold_fit for c in
             visualizer.gcm_rcm_couples])

    def get_function_from_fit(self, ensemble_fit, massif_name, gcm_rcm_couple):
        visualizer = ensemble_fit.gcm_rcm_couple_to_visualizer[gcm_rcm_couple]
        one_fold_fit = visualizer.massif_name_to_one_fold_fit[massif_name]
        return one_fold_fit.best_function_from_fit
