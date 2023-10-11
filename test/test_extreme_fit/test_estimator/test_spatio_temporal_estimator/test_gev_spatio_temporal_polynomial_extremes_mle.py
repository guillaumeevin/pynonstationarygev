import unittest

import numpy as np
import pandas as pd

from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from extreme_fit.model.utils import r, set_seed_r
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class TestGevTemporalQuadraticExtremesMle(unittest.TestCase):

    def setUp(self) -> None:
        nb_data = 100
        set_seed_r()
        r("""
        N <- {}
        loc = 0; scale = 1; shape <- 0.1
        x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
        start_loc = 0; start_scale = 1; start_shape = 1
        """.format(nb_data))

        # Compute coordinates
        altitudes = [300, 600]
        temporal_coordinates = ConsecutiveTemporalCoordinates.from_nb_temporal_steps(nb_data)
        spatial_coordinates = AbstractSpatialCoordinates.from_list_x_coordinates(altitudes)
        self.coordinates = AbstractSpatioTemporalCoordinates.from_spatial_coordinates_and_temporal_coordinates(
            spatial_coordinates,
            temporal_coordinates)

        # Compute observations
        a = np.array(r['x_gev'])
        data = np.concatenate([a, a], axis=0)
        df2 = pd.DataFrame(data=data, index=self.coordinates.index)
        observations = AbstractSpatioTemporalObservations(df_maxima_gev=df2)

        self.dataset = AbstractDataset(observations=observations, coordinates=self.coordinates)
        self.fit_method = MarginFitMethod.extremes_fevd_mle




if __name__ == '__main__':
    unittest.main()
