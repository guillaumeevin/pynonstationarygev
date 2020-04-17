from typing import Tuple, Dict, List

from cached_property import cached_property

from projects.contrasting_trends_in_snow_loads.altitudes_fit.altitudes_studies import AltitudesStudies
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.slicer.split import invert_s_split


class TwoFoldDatasetsGenerator(object):

    def __init__(self, studies: AltitudesStudies, nb_samples, massif_names=None):
        self.studies = studies
        self.nb_samples = nb_samples
        if massif_names is None:
            self.massif_names = self.studies.study.all_massif_names()
        else:
            self.massif_names = massif_names

    @cached_property
    def massif_name_to_list_two_fold_datasets(self) -> Dict[str, List[Tuple[AbstractDataset, AbstractDataset]]]:
        d = {}
        for massif_name in self.massif_names:
            l = []
            for _ in range(self.nb_samples):
                # Append to the list
                l.append(self.two_fold_datasets(massif_name))
            d[massif_name] = l
        return d

    def two_fold_datasets(self, massif_name: str) -> Tuple[AbstractDataset, AbstractDataset]:
        # Create split for the 1st fold
        s_split_temporal = self.studies.random_s_split_temporal(train_split_ratio=0.5)
        dataset_fold_1 = self.studies.spatio_temporal_dataset(massif_name=massif_name,
                                                              s_split_temporal=s_split_temporal)
        # Invert the s_split for the 2nd fold
        s_split_temporal_inverted = invert_s_split(s_split_temporal)
        dataset_fold_2 = self.studies.spatio_temporal_dataset(massif_name=massif_name,
                                                              s_split_temporal=s_split_temporal_inverted)
        return dataset_fold_1, dataset_fold_2
