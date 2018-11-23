import unittest

from extreme_estimator.gev_params import GevParams
from extreme_estimator.extreme_models.margin_model.smooth_margin_model import LinearShapeAxis0MarginModel
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import CircleCoordinates


class TestLinearMarginModel(unittest.TestCase):
    DISPLAY = False

    def test_visualization_2D(self):
        spatial_coordinates = CircleCoordinates.from_nb_points(nb_points=50)
        margin_model = LinearShapeAxis0MarginModel(coordinates=spatial_coordinates)
        for gev_param_name in GevParams.GEV_PARAM_NAMES:
            margin_model.margin_function_sample.visualize_2D(gev_param_name=gev_param_name, show=self.DISPLAY)
        # maxima_gev = margin_model.rmargin_from_nb_obs(nb_obs=10, coordinates=coordinates)
        # fitted_margin_function = margin_model.fitmargin_from_maxima_gev(maxima_gev=maxima_gev, coordinates=coordinates)


if __name__ == '__main__':
    unittest.main()
