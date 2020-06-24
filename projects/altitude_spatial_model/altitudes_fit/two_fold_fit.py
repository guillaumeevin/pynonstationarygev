from typing import Dict, List

from cached_property import cached_property

from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from projects.altitude_spatial_model.altitudes_fit.two_fold_datasets_generator import TwoFoldDatasetsGenerator
from projects.altitude_spatial_model.altitudes_fit.two_fold_detail_fit import TwoFoldMassifFit
from projects.altitude_spatial_model.altitudes_fit.utils import Score, Grouping


class TwoFoldFit(object):

    def __init__(self, two_fold_datasets_generator: TwoFoldDatasetsGenerator,
                 model_family_name_to_model_classes: Dict[str, List[type]],
                 fit_method=MarginFitMethod.extremes_fevd_mle):
        self.two_fold_datasets_generator = two_fold_datasets_generator
        self.fit_method = fit_method
        self.model_family_name_to_model_classes = model_family_name_to_model_classes

        self.massif_name_to_massif_fit = {}
        for massif_name, list_two_fold_datasets in self.two_fold_datasets_generator.massif_name_to_list_two_fold_datasets.items():
            self.massif_name_to_massif_fit[massif_name] = TwoFoldMassifFit(model_classes=self.model_classes_to_fit,
                                                                           list_two_fold_datasets=list_two_fold_datasets,
                                                                           fit_method=self.fit_method)

    @cached_property
    def model_classes_to_fit(self):
        return set().union(*[set(model_classes) for model_classes in self.model_family_name_to_model_classes.values()])

    def model_family_name_to_best_model(self, score):
        pass

    def massif_name_to_best_model(self, score=Score.NLLH_TEST, group=Grouping.MEAN_RANKING):
        return {
            massif_name: massif_fit.best_model(score, group)
            for massif_name, massif_fit in self.massif_name_to_massif_fit.items()
        }
