import pandas as pd

from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates


class ConsecutiveTemporalCoordinates(AbstractTemporalCoordinates):
    pass

    @classmethod
    def from_nb_temporal_steps(cls, nb_temporal_steps, start=0, end=None):
        df = cls.df_temporal(nb_temporal_steps, start)
        if end is not None:
            df /= df[cls.COORDINATE_T].max()
            if end > 0:
                df *= end
        return cls.from_df(df)

    @classmethod
    def df_temporal(cls, nb_temporal_steps, start=0):
        df = pd.DataFrame.from_dict({cls.COORDINATE_T: list(range(start, start + nb_temporal_steps))})
        return df
