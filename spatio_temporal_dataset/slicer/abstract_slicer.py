from typing import Union, List

import pandas as pd

from spatio_temporal_dataset.slicer.split import Split


class AbstractSlicer(object):

    def __init__(self, coordinates_train_ind: Union[None, pd.Series], observations_train_ind: Union[None, pd.Series]):
        self.index_train_ind = coordinates_train_ind  # type: Union[None, pd.Series]
        self.column_train_ind = observations_train_ind  # type: Union[None, pd.Series]

    @property
    def train_split(self) -> Split:
        pass

    @property
    def test_split(self) -> Split:
        pass

    @property
    def splits(self) -> List[Split]:
        pass


    @property
    def index_test_ind(self) -> pd.Series:
        return ~self.index_train_ind

    # todo: test should be the same as train when we don't care about that in the split
    @property
    def column_test_ind(self) -> pd.Series:
        return ~self.column_train_ind

    @property
    def some_required_ind_are_not_defined(self):
        pass

    def summary(self):
        print('Slicer summary: \n')
        for s, global_name in [(self.index_train_ind, "Spatial"), (self.column_train_ind, "Temporal")]:
            print(global_name + ' split')
            if s is None:
                print('Not handled by this slicer')
            else:
                for f, name in [(len, 'Total'), (sum, 'train')]:
                    print("{}: {}".format(name, f(s)))
                print('\n')

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

    def specialized_loc_split(self, df: pd.DataFrame, split: Split):
        # This method should be defined in the child class
        return None


def slice(df: pd.DataFrame, split: Split = Split.all, slicer: AbstractSlicer = None) -> pd.DataFrame:
    if slicer is None:
        assert split is Split.all
        return df
    else:
        return slicer.loc_split(df, split)
