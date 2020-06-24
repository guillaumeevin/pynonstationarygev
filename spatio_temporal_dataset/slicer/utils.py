from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.slicer.spatio_temporal_slicer import SpatioTemporalSlicer


def get_slicer_class_from_s_splits(s_split_spatial, s_split_temporal):
    if s_split_temporal is None and s_split_spatial is None:
        return SpatioTemporalSlicer
    else:
        return AbstractCoordinates.slicer_class_from_s_splits(s_split_spatial=s_split_spatial,
                                                          s_split_temporal=s_split_temporal)
