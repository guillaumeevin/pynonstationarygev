import unittest
from collections import OrderedDict

import numpy as np
import pandas as pd

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import UniformSpatialCoordinates, \
    LinSpaceSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import \
    CircleSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.generated_spatio_temporal_coordinates import \
    GeneratedSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate
from spatio_temporal_dataset.coordinates.utils import get_index_with_spatio_temporal_index_suffix
from test.test_utils import load_test_temporal_coordinates


class TestSpatialCoordinates(unittest.TestCase):
    DISPLAY = False

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.coord = None  # type:  AbstractCoordinates

    def tearDown(self):
        if self.DISPLAY:
            self.coord.visualize()
        first_coordinate = self.coord.df_all_coordinates.iloc[0, 0]
        self.assertFalse(np.isnan(first_coordinate))

    def test_unif(self):
        self.coord = UniformSpatialCoordinates.from_nb_points(nb_points=10)

    def test_circle(self):
        self.coord = CircleSpatialCoordinates.from_nb_points(nb_points=500)



class SpatioTemporalCoordinates(unittest.TestCase):
    nb_points = 4
    nb_steps = 2

    def test_unique_spatio_temporal_index_and_matching_spatial_index(self):
        spatial_coordinates = LinSpaceSpatialCoordinates.from_nb_points(self.nb_points)
        spatial_indexes = [[10, 11, 12, 13], ['a', 'b', 'c', 'd']]
        for spatial_index in spatial_indexes:
            spatial_coordinates.df_all_coordinates.index = spatial_index
            df_spatial = spatial_coordinates.df_spatial_coordinates()
            coordinates = GeneratedSpatioTemporalCoordinates.from_df_spatial_and_nb_steps(df_spatial=df_spatial,
                                                                                          nb_steps=self.nb_steps)

            # the uniqueness of each spatio temporal index is not garanteed by the current algo
            # it will work in classical cases, and raise an assert when uniqueness is needed (when using a slicer)
            index1 = pd.Series(spatial_coordinates.spatial_index)
            index2 = pd.Series(coordinates.spatial_index)
            ind = index1 != index2  # type: pd.Series
            self.assertEqual(sum(ind), 0, msg="spatial_coordinates:\n{} \n!= spatio_temporal_coordinates \n{}".
                             format(index1.loc[ind], index2.loc[ind]))

            index1 = get_index_with_spatio_temporal_index_suffix(spatial_coordinates.df_spatial_coordinates(), t=0)
            index1 = pd.Series(index1)
            index2 = pd.Series(coordinates.df_spatial_coordinates().index)
            ind = index1 != index2  # type: pd.Series
            self.assertEqual(sum(ind), 0, msg="spatial_coordinates:\n{} \n!= spatio_temporal_coordinates \n{}".
                             format(index1.loc[ind], index2.loc[ind]))

    def test_ordered_coordinates(self):
        # Order coordinates, to ensure that the first dimension/the second dimension and so on..
        # Always are in the same order to a given type (e.g. spatio_temporal= of coordinates
        # Check space coordinates
        d = OrderedDict()
        d[AbstractCoordinates.COORDINATE_Z] = [1]
        d[AbstractCoordinates.COORDINATE_X] = [1]
        d[AbstractCoordinates.COORDINATE_Y] = [1]
        df = pd.DataFrame.from_dict(d)
        for df2 in [df, df.loc[:, ::-1]][-1:]:
            coordinates = AbstractCoordinates(df=df2)
            self.assertEqual(list(coordinates.df_all_coordinates.columns),
                             [AbstractCoordinates.COORDINATE_X, AbstractCoordinates.COORDINATE_Y,
                              AbstractCoordinates.COORDINATE_Z])
        # Check space/time ordering
        d = OrderedDict()
        d[AbstractCoordinates.COORDINATE_T] = [1]
        d[AbstractCoordinates.COORDINATE_X] = [1]
        df = pd.DataFrame.from_dict(d)
        for df2 in [df, df.loc[:, ::-1]][-1:]:
            coordinates = AbstractCoordinates(df=df2)
            self.assertEqual(list(coordinates.df_all_coordinates.columns),
                             [AbstractCoordinates.COORDINATE_X, AbstractCoordinates.COORDINATE_T])



class TestCoordinatesWithModifiedCovariate(unittest.TestCase):

    def test_time_covariate(self):
        coordinates = load_test_temporal_coordinates(nb_steps=10)[0]
        old_df = coordinates.df_temporal_coordinates_for_fit().copy()
        new_df = coordinates.df_temporal_coordinates_for_fit(temporal_covariate_for_fit=TimeTemporalCovariate)
        pd.testing.assert_frame_equal(old_df, new_df)


if __name__ == '__main__':
    unittest.main()
