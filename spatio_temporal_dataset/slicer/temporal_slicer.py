from typing import List, Union

import pandas as pd

from spatio_temporal_dataset.slicer.abstract_slicer import AbstractSlicer
from spatio_temporal_dataset.slicer.split import Split


class TemporalSlicer(AbstractSlicer):
    SPLITS = [Split.train_temporal, Split.test_temporal]

    def __init__(self, ind_train_temporal: Union[None, pd.Series]):
        super().__init__(None, ind_train_temporal)

    @property
    def splits(self) -> List[Split]:
        return self.SPLITS

    @property
    def train_split(self) -> Split:
        return Split.train_temporal

    @property
    def test_split(self) -> Split:
        return Split.test_temporal

    @property
    def some_required_ind_are_not_defined(self) -> bool:
        return self.ind_train_temporal is None

    def specialized_loc_split(self, df: pd.DataFrame, split: Split) -> pd.DataFrame:
        assert pd.Index.equals(df.index, self.ind_train_temporal.index)
        if split is Split.train_temporal:
            return df.loc[self.ind_train_temporal]
        elif split is Split.test_temporal:
            return df.loc[self.ind_test_temporal]
