from typing import List

from projects.quantile_regression_vs_evt.AbstractSimulation import AbstractSimulation
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import MarginAnnualMaxima


class GevSimulation(AbstractSimulation):

    def __init__(self, margin_model_class_for_simulation, nb_time_series, quantile, time_series_lengths=None, multiprocessing=False, model_classes=None):
        super().__init__(nb_time_series, quantile, time_series_lengths, multiprocessing, model_classes)
        self.margin_model_class_for_simulation = margin_model_class_for_simulation

    def generate_all_observation(self, nb_time_series, length, coordinates) -> List[AbstractSpatioTemporalObservations]:
        margin_model = self.margin_model_class_for_simulation(coordinates)
        return [MarginAnnualMaxima.from_sampling(nb_obs=length, coordinates=coordinates, margin_model=margin_model) for _ in range(nb_time_series)]

