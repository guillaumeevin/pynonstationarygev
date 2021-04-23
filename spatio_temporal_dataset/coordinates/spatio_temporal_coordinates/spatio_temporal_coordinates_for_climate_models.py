import pandas as pd

from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates


class SpatioTemporalCoordinatesForClimateModels(AbstractSpatioTemporalCoordinates):

    def __init__(self, df: pd.DataFrame = None,
                 transformation_class: type = None, spatial_coordinates: AbstractSpatialCoordinates = None,
                 temporal_coordinates: AbstractTemporalCoordinates = None,
                 gcm_rcm_couple=None,
                 scenario_str=None):
        df = self.load_df_is_needed(df, spatial_coordinates, temporal_coordinates)
        # Assign the climate model coordinates
        gcm, rcm = gcm_rcm_couple
        df[self.COORDINATE_RCP] = scenario_str
        df[self.COORDINATE_GCM] = gcm
        df[self.COORDINATE_RCM] = rcm
        super().__init__(df, transformation_class, spatial_coordinates, temporal_coordinates)
