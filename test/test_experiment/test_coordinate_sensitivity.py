import unittest

from experiment.meteo_france_SCM_study.crocus.crocus import CrocusSwe
from experiment.meteo_france_SCM_study.visualization.study_visualization.main_study_visualizer import \
    study_iterator_global
from experiment.trend_analysis.non_stationary_trends import \
    ConditionalIndedendenceLocationTrendTest
from experiment.meteo_france_SCM_study.visualization.study_visualization.study_visualizer import StudyVisualizer
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization, BetweenZeroAndOneNormalizationMinEpsilon, BetweenZeroAndOneNormalizationMaxEpsilon
from utils import get_display_name_from_object_type


class TestCoordinateSensitivity(unittest.TestCase):
    DISPLAY = False

    def test_coordinate_normalization_sensitivity(self):
        altitudes = [300, 600, 900, 1200, 2100, 3000][-1:]
        transformation_classes = [None, BetweenZeroAndOneNormalization, BetweenZeroAndOneNormalizationMinEpsilon,
                                  BetweenZeroAndOneNormalizationMaxEpsilon][1:2]

        study_classes = [CrocusSwe]
        for study in study_iterator_global(study_classes, altitudes=altitudes, verbose=False):
            if self.DISPLAY:
                print(study.altitude)
            for transformation_class in transformation_classes:
                study_visualizer = StudyVisualizer(study, transformation_class=transformation_class)
                study_visualizer.temporal_non_stationarity = True
                trend_test = ConditionalIndedendenceLocationTrendTest(study_visualizer.dataset)
                years = [1960, 1990]
                mu1s = [trend_test.get_mu_coefs(year)['mu_temporal'] for year in years]
                if self.DISPLAY:
                    print('Stationary')
                    stationary_est = trend_test.load_estimator(trend_test.stationary_margin_model_class,
                                                               starting_point=None)
                    print(stationary_est.result_from_fit.convergence)
                    print(stationary_est.margin_function_fitted.coef_dict)
                    print('Non Stationary')
                    non_stationary_est = trend_test.load_estimator(trend_test.non_stationary_margin_model_class,
                                                                   starting_point=1960)
                    print(non_stationary_est.result_from_fit.convergence)
                    non_stationary_est = trend_test.load_estimator(trend_test.non_stationary_margin_model_class,
                                                                   starting_point=1990)
                    print(non_stationary_est.result_from_fit.convergence)
                    print(non_stationary_est.margin_function_fitted.coef_dict)
                    print(get_display_name_from_object_type(transformation_class), 'mu1s: ', mu1s)
                    print('\n')
                self.assertTrue(0.0 not in mu1s)


if __name__ == '__main__':
    unittest.main()
