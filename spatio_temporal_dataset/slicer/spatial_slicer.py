from typing import List, Union

import pandas as pd

from spatio_temporal_dataset.slicer.abstract_slicer import AbstractSlicer
from spatio_temporal_dataset.slicer.split import Split


class SpatialSlicer(AbstractSlicer):
    SPLITS = [Split.train_spatial, Split.test_spatial]

    def __init__(self, ind_train_spatial: Union[None, pd.Series]):
        super().__init__(ind_train_spatial, None)

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
    def some_required_ind_are_not_defined(self) -> bool:
        return self.ind_train_spatial is None

    def specialized_loc_split(self, df: pd.DataFrame, split: Split) -> pd.DataFrame:
        assert pd.Index.equals(df.index, self.ind_train_spatial.index)
        if split is Split.train_spatial:
            return df.loc[self.ind_train_spatial]
        elif split is Split.test_spatial:
            return df.loc[self.ind_test_spatial]
