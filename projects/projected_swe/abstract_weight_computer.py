from collections import OrderedDict
import pandas as pd

import numpy as np
from scipy.special import softmax

from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_str
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import compute_nllh
from extreme_trend.ensemble_fit.independent_ensemble_fit.independent_ensemble_fit import IndependentEnsembleFit
from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble
from projects.projected_swe.utils import WEIGHT_COLUMN_NAME, get_csv_filepath, save_to_filepath
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate
from collections import OrderedDict

import numpy as np
from scipy.special import softmax

from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import compute_nllh
from extreme_trend.ensemble_fit.independent_ensemble_fit.independent_ensemble_fit import IndependentEnsembleFit
from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate


class AbstractWeightComputer(object):

    def __init__(self, visualizer: VisualizerForProjectionEnsemble, scm_study_class, year_min, year_max):
        self.scm_study_class = scm_study_class
        self.year_max = year_max
        self.year_min = year_min
        self.visualizer = visualizer

    def compute_weights_and_save_them(self):
        column_names = [WEIGHT_COLUMN_NAME] + self.visualizer.gcm_rcm_couples
        df = pd.DataFrame(index=self.visualizer.gcm_rcm_couples, columns=column_names)
        for i, column_name in enumerate(column_names):
            if i == 0:
                gcm_rcm_couples_subset = self.visualizer.gcm_rcm_couples
                gcm_rcm_couple_missing = None
            else:
                index_missing = i - 1
                gcm_rcm_couple_missing = self.visualizer.gcm_rcm_couples[index_missing]
                gcm_rcm_couples_subset = self.visualizer.gcm_rcm_couples[:]
                gcm_rcm_couples_subset.remove(gcm_rcm_couple_missing)
            # Compute weights
            gcm_rcm_couple_to_weight = self.compute_weight(gcm_rcm_couples_subset, gcm_rcm_couple_missing)
            column_values = [gcm_rcm_couple_to_weight[c] for c in df.index]
            df[column_name] = column_values

        # Save csv
        save_to_filepath(df, self.visualizer.gcm_rcm_couples, self.visualizer.altitudes_list,
                         self.year_min, self.year_max,
                         self.visualizer.scenario)

    def compute_weight(self, gcm_rcm_couples_subset, gcm_rcm_couple_missing):
        # Initial the dictionary
        gcm_rcm_couple_to_nllh = OrderedDict()
        for gcm_rcm_couple in gcm_rcm_couples_subset:
            gcm_rcm_couple_to_nllh[gcm_rcm_couple] = 0
        for ensemble_fit in self.visualizer.ensemble_fits(ensemble_class=IndependentEnsembleFit):
            # Load the AltitudeStudies
            if gcm_rcm_couple_missing is None:
                reference_studies = AltitudesStudies(self.scm_study_class, ensemble_fit.altitudes,
                                                     year_min=self.year_min,
                                                     year_max=self.year_max)
            else:
                reference_studies = ensemble_fit.gcm_rcm_couple_to_visualizer[gcm_rcm_couple_missing].studies
            # Computer the nllh for each "station", i.e. each altitude and massif_name
            for altitude, reference_study in reference_studies.altitude_to_study.items():
                for massif_name, maxima in reference_study.massif_name_to_annual_maxima.items():
                    assert all([self.year_min <= year <= self.year_max for year in reference_study.ordered_years])
                    gcm_rcm_couple_to_local_nllh = self.compute_gcm_rcm_couple_to_local_nllh(ensemble_fit,
                                                                                             gcm_rcm_couples_subset,
                                                                                             massif_name,
                                                                                             maxima,
                                                                                             altitude,
                                                                                             reference_study)
                    if gcm_rcm_couple_to_local_nllh is not None:
                        for c, nllh in gcm_rcm_couple_to_local_nllh.items():
                            gcm_rcm_couple_to_nllh[c] += nllh
        # Compute weights
        weights = softmax(-np.array(list(gcm_rcm_couple_to_nllh.values())))
        weights = [w[0] if isinstance(w, np.ndarray) else w for w in weights]
        gcm_rcm_couple_to_weight = dict(zip(gcm_rcm_couples_subset, weights))
        # Add missing weight
        gcm_rcm_couple_to_weight[gcm_rcm_couple_missing] = np.nan
        return gcm_rcm_couple_to_weight

    def compute_gcm_rcm_couple_to_local_nllh(self, ensemble_fit, gcm_rcm_couples_subset, massif_name, maxima,
                                             altitude, reference_study):
        raise NotImplementedError
