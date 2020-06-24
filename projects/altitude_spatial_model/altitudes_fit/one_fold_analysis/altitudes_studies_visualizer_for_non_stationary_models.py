from typing import List

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import \
    OneFoldFit


class AltitudesStudiesVisualizerForNonStationaryModels(StudyVisualizer):

    def __init__(self, studies: AltitudesStudies,
                 model_classes: List[AbstractSpatioTemporalPolynomialModel],
                 show=False,
                 massif_names=None):
        study = studies.study
        self.massif_names = massif_names if massif_names is not None else self.study.study_massif_names
        self.studies = studies
        self.non_stationary_models = model_classes
        super().__init__(study, show=show, save_to_file=not show)
        self.massif_name_to_one_fold_fit = {}
        for massif_name in self.massif_names:
            dataset = studies.spatio_temporal_dataset(massif_name=massif_name)
            old_fold_fit = OneFoldFit(dataset, model_classes)
            self.massif_name_to_one_fold_fit[massif_name] = old_fold_fit