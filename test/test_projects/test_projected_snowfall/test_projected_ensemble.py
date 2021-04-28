

import unittest
from typing import List

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, get_gcm_rcm_couples
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_snowf import SafranSnowfall2019
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryShapeTemporalModel
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearScaleAndShapeOneLinearLocModel, NonStationaryTwoLinearLocationAndScaleAndShapeModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.quantile_model.quantile_regression_model import ConstantQuantileRegressionModel
from extreme_fit.model.utils import set_seed_for_test
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble
from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from projects.archive.quantile_regression_vs_evt.annual_maxima_simulation.gev_simulation import StationarySimulation
from projects.projected_extreme_snowfall.results.part_3.main_projections_ensemble import set_up_and_load
from projects.projected_extreme_snowfall.results.utils import SPLINE_MODELS_FOR_PROJECTION_ONE_ALTITUDE
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate


class TestProjectedEnsemble(unittest.TestCase):
    DISPLAY = False

    def test_projected_ensemble(self):
        study_class = AdamontSnowfall
        massif_names = ['Vanoise']
        temporal_covariate_for_fit = AnomalyTemperatureWithSplineTemporalCovariate
        set_seed_for_test()
        scenario = AdamontScenario.rcp85_extended
        AltitudesStudiesVisualizerForNonStationaryModels.consider_at_least_two_altitudes = False
        gcm_rcm_couples = get_gcm_rcm_couples(scenario)
        gcm_rcm_couples = gcm_rcm_couples[:2] + gcm_rcm_couples[-2:]
        altitudes_list = [[2700]]
        model_classes = [StationaryTemporalModel,
                         NonStationaryShapeTemporalModel,
                         NonStationaryTwoLinearScaleAndShapeOneLinearLocModel,
                         NonStationaryTwoLinearLocationAndScaleAndShapeModel]

        climate_coordinates_with_effects_list = [None,
                                                 [AbstractCoordinates.COORDINATE_GCM,
                                                  AbstractCoordinates.COORDINATE_RCM],
                                                 [AbstractCoordinates.COORDINATE_GCM],
                                                 [AbstractCoordinates.COORDINATE_RCM],
                                                 ]  # None means we do not create any effect

        # Default parameters
        gcm_to_year_min_and_year_max = None
        only_model_that_pass_gof = False
        remove_physically_implausible_models = False
        safran_study_class = SafranSnowfall2019
        ensemble_fit_classes = [TogetherEnsembleFit]

        idx_list = [(0,0,0), (0, 1, 0), (0, 2, 3), (1, 2, 3)]
        for i1, i2, i3 in idx_list:
            param_name_to_climate_coordinates_with_effects = {
                GevParams.LOC: climate_coordinates_with_effects_list[i1],
                GevParams.SCALE: climate_coordinates_with_effects_list[i2],
                GevParams.SHAPE: climate_coordinates_with_effects_list[i3],
            }
            print(param_name_to_climate_coordinates_with_effects)

            visualizer = VisualizerForProjectionEnsemble(
                altitudes_list, gcm_rcm_couples, study_class, Season.annual, scenario,
                model_classes=model_classes,
                ensemble_fit_classes=ensemble_fit_classes,
                massif_names=massif_names,
                fit_method=MarginFitMethod.evgam,
                temporal_covariate_for_fit=temporal_covariate_for_fit,
                remove_physically_implausible_models=remove_physically_implausible_models,
                gcm_to_year_min_and_year_max=gcm_to_year_min_and_year_max,
                safran_study_class=safran_study_class,
                display_only_model_that_pass_gof_test=only_model_that_pass_gof,
                param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
            )
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
