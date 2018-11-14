import unittest

from extreme_estimator.R_model.gev.gev_parameters import GevParams
from extreme_estimator.R_model.margin_function.independent_margin_function import LinearMarginFunction
from extreme_estimator.R_model.margin_model.smooth_margin_model import ConstantMarginModel, LinearShapeAxis0MarginModel
from spatio_temporal_dataset.spatial_coordinates.generated_coordinates import CircleCoordinatesRadius1


class TestLinearMarginModel(unittest.TestCase):
    DISPLAY = True

    def test_visualization_2D(self):
        spatial_coordinates = CircleCoordinatesRadius1.from_nb_points(nb_points=50)
        margin_model = LinearShapeAxis0MarginModel(spatial_coordinates=spatial_coordinates)
        for gev_param_name in GevParams.GEV_PARAM_NAMES:
            margin_model.margin_function_sample.visualize_2D(gev_param_name=gev_param_name, show=self.DISPLAY)
        # maxima_gev = margin_model.rmargin_from_nb_obs(nb_obs=10, coordinates=coordinates)
        # fitted_margin_function = margin_model.fitmargin_from_maxima_gev(maxima_gev=maxima_gev, coordinates=coordinates)


if __name__ == '__main__':
    unittest.main()
