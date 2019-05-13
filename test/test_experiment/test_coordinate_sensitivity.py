import unittest

from experiment.meteo_france_SCM_study.crocus.crocus import CrocusSwe
from experiment.meteo_france_SCM_study.visualization.study_visualization.main_study_visualizer import \
    study_iterator_global
from experiment.meteo_france_SCM_study.visualization.study_visualization.non_stationary_trends import \
    ConditionalIndedendenceLocationTrendTest
from experiment.meteo_france_SCM_study.visualization.study_visualization.study_visualizer import StudyVisualizer
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization, BetweenMinusOneAndOneNormalization
from utils import get_display_name_from_object_type


class TestCoordinateSensitivity(unittest.TestCase):
    DISPLAY = False

    def test_coordinate_normalization_sensitivity(self):
        altitudes = [3000]
        transformation_classes = [BetweenZeroAndOneNormalization, BetweenMinusOneAndOneNormalization][:]
        for transformation_class in transformation_classes:
            study_classes = [CrocusSwe]
            for study in study_iterator_global(study_classes, altitudes=altitudes, verbose=False):
                study_visualizer = StudyVisualizer(study, transformation_class=transformation_class)
                study_visualizer.temporal_non_stationarity = True
                trend_test = ConditionalIndedendenceLocationTrendTest(study_visualizer.dataset)
                years = [1960, 1990]
                mu1s = [trend_test.get_mu1(year) for year in years]
                if self.DISPLAY:
                    print('Stationary')
                    print(trend_test.get_estimator(trend_test.stationary_margin_model_class, starting_point=None).margin_function_fitted.coef_dict)
                    print('Non Stationary')
                    print(trend_test.get_estimator(trend_test.non_stationary_margin_model_class, starting_point=1960).margin_function_fitted.coef_dict)
                    print(get_display_name_from_object_type(transformation_class), 'mu1s: ', mu1s)
                self.assertTrue(0.0 not in mu1s)


if __name__ == '__main__':
    unittest.main()
