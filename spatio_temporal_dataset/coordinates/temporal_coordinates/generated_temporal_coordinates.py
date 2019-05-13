import pandas as pd

from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates


class ConsecutiveTemporalCoordinates(AbstractTemporalCoordinates):
    pass

    @classmethod
    def from_nb_temporal_steps(cls, nb_temporal_steps, train_split_ratio: float = None, start=0,
                               transformation_class: type = None):
        df = cls.df_temporal(nb_temporal_steps, start)
        return cls.from_df(df, train_split_ratio, transformation_class=transformation_class)

    @classmethod
    def df_temporal(cls, nb_temporal_steps, start=0):
        df = pd.DataFrame.from_dict({cls.COORDINATE_T: list(range(start, start + nb_temporal_steps))})
        return df
