from typing import List, Union

import pandas as pd

from spatio_temporal_dataset.slicer.abstract_slicer import AbstractSlicer
from spatio_temporal_dataset.slicer.split import Split


class SpatialSlicer(AbstractSlicer):
    SPLITS = [Split.train_spatial, Split.test_spatial]

    def __init__(self, coordinates_train_ind: Union[None, pd.Series], observations_train_ind: Union[None, pd.Series]):
        super().__init__(coordinates_train_ind, None)

    @property
    def splits(self) -> List[Split]:
        return self.SPLITS

    @property
    def train_split(self) -> Split:
        return Split.train_spatial

    @property
    def test_split(self) -> Split:
        return Split.test_spatial

    @property
    def some_required_ind_are_not_defined(self):
        return self.index_train_ind is None

    def specialized_loc_split(self, df: pd.DataFrame, split: Split):
        assert pd.Index.equals(df.index, self.index_train_ind.index)
        if split is Split.train_spatial:
            return df.loc[self.index_train_ind, :]
        elif split is Split.test_spatial:
            return df.loc[self.index_test_ind, :]
