from typing import Union, List

import pandas as pd

from spatio_temporal_dataset.slicer.split import Split


class AbstractSlicer(object):

    def __init__(self, ind_train_spatial: Union[None, pd.Series], ind_train_temporal: Union[None, pd.Series]):
        self.ind_train_spatial = ind_train_spatial  # type: Union[None, pd.Series]
        self.ind_train_temporal = ind_train_temporal  # type: Union[None, pd.Series]

    @property
    def ind_test_spatial(self) -> pd.Series:
        return ~self.ind_train_spatial

    @property
    def ind_test_temporal(self) -> pd.Series:
        return ~self.ind_train_temporal

    def loc_split(self, df: pd.DataFrame, split: Split):
        # split should belong to the list of split accepted by the slicer
        assert isinstance(split, Split)

        if split is Split.all:
            return df

        assert split in self.splits, "split:{}, slicer_type:{}".format(split, type(self))

        # By default, some required splits are not defined
        # instead of crashing, we return all the data for all the split
        # This is the default behavior, when the required splits has been defined
        if self.some_required_ind_are_not_defined:
            return df
        else:
            return self.specialized_loc_split(df=df, split=split)

    def summary(self, show=True):
        msg = ''
        for s, global_name in [(self.ind_train_spatial, "Spatial"), (self.ind_train_temporal, "Temporal")]:
            msg += global_name + ': '
            if s is None:
                msg += 'Not handled by this slicer'
            else:
                for f, name in [(len, 'Total'), (sum, 'train')]:
                    msg += "{}: {} ".format(name, f(s))
                msg += ' / '
        if show:
            print(msg)
        return msg

    # Methods that need to be defined in the child class

    def specialized_loc_split(self, df: pd.DataFrame, split: Split):
        return None

    @property
    def some_required_ind_are_not_defined(self):
        pass

    @property
    def train_split(self) -> Split:
        pass

    @property
    def test_split(self) -> Split:
        pass

    @property
    def splits(self) -> List[Split]:
        pass


def df_sliced(df: pd.DataFrame, split: Split = Split.all, slicer: AbstractSlicer = None) -> pd.DataFrame:
    if slicer is None:
        assert split is Split.all
        return df
    else:
        return slicer.loc_split(df, split)
