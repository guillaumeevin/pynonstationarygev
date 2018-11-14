import unittest

from extreme_estimator.R_model.margin_model.smooth_margin_model import ConstantMarginModel
from spatio_temporal_dataset.spatial_coordinates.generated_coordinates import CircleCoordinatesRadius1


class TestMarginModel(unittest.TestCase):
    DISPLAY = True
    MARGIN_TYPES = [ConstantMarginModel]

    # def test_visualization(self):
    #     coord_2D = CircleCoordinatesRadius1.from_nb_points(nb_points=50)
    #     if self.DISPLAY:
    #         coord_2D.visualization_2D()
    #     for margin_class in self.MARGIN_TYPES:
    #         margin_model = margin_class()
    #         margin_model.visualize(coordinates=coord_2D.coordinates)


if __name__ == '__main__':
    unittest.main()
