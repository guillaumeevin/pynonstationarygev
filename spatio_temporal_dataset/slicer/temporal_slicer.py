from typing import List, Union

import pandas as pd

from spatio_temporal_dataset.slicer.abstract_slicer import AbstractSlicer
from spatio_temporal_dataset.slicer.split import Split


class TemporalSlicer(AbstractSlicer):
    SPLITS = [Split.train_temporal, Split.test_temporal]

    def __init__(self, coordinates_train_ind: Union[None, pd.Series], observations_train_ind: Union[None, pd.Series]):
        super().__init__(None, observations_train_ind)

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
    def some_required_ind_are_not_defined(self):
        return self.column_train_ind is None

    def specialized_loc_split(self, df: pd.DataFrame, split: Split):
        assert pd.Index.equals(df.columns, self.column_train_ind.index)
        if split is Split.train_temporal:
            return df.loc[:, self.column_train_ind]
        elif split is Split.test_temporal:
            return df.loc[:, self.column_test_ind]
