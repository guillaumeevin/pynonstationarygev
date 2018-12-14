from typing import List

import pandas as pd

from spatio_temporal_dataset.slicer.abstract_slicer import AbstractSlicer
from spatio_temporal_dataset.slicer.split import Split


class SpatioTemporalSlicer(AbstractSlicer):
    SPLITS = [Split.train_spatiotemporal,
              Split.test_spatiotemporal,
              Split.test_spatiotemporal_spatial,
              Split.test_spatiotemporal_temporal]

    @property
    def splits(self) -> List[Split]:
        return self.SPLITS

    @property
    def train_split(self) -> Split:
        return Split.train_spatiotemporal

    @property
    def test_split(self) -> Split:
        return Split.test_spatiotemporal

    @property
    def some_required_ind_are_not_defined(self) -> bool:
        return self.ind_train_spatial is None or self.ind_train_temporal is None

    def specialized_loc_split(self, df: pd.DataFrame, split: Split) -> pd.DataFrame:
        assert pd.Index.equals(df.index, self.ind_train_temporal.index)
        assert pd.Index.equals(df.index, self.ind_train_spatial.index)
        if split is Split.train_spatiotemporal:
            return df.loc[self.ind_train_spatial & self.ind_train_temporal]
        elif split is Split.test_spatiotemporal:
            return df.loc[self.ind_test_spatial & self.ind_test_temporal]
        elif split is Split.test_spatiotemporal_spatial:
            return df.loc[self.ind_test_spatial & self.ind_train_temporal]
        elif split is Split.test_spatiotemporal_temporal:
            return df.loc[self.ind_train_spatial & self.ind_test_temporal]
