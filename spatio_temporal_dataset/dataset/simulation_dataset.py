from extreme_estimator.extreme_models.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import AbstractMaxStableModel
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.temporal_observations.abstract_temporal_observations import AbstractTemporalObservations
from spatio_temporal_dataset.temporal_observations.annual_maxima_observations import \
    MaxStableAnnualMaxima, AnnualMaxima, MarginAnnualMaxima, FullAnnualMaxima


class SimulatedDataset(AbstractDataset):
    """
    Class SimulatedDataset gives access to:
        -the max_stable_model AND/OR marginal_model that was used for sampling
    """

    def __init__(self, temporal_observations: AbstractTemporalObservations,
                 coordinates: AbstractCoordinates,
                 max_stable_model: AbstractMaxStableModel = None,
                 margin_model: AbstractMarginModel = None):
        super().__init__(temporal_observations, coordinates)
        assert margin_model is not None or max_stable_model is not None
        self.margin_model = margin_model  # type: AbstractMarginModel
        self.max_stable_model = max_stable_model  #  type: AbstractMaxStableModel


class MaxStableDataset(SimulatedDataset):

    @classmethod
    def from_sampling(cls, nb_obs: int, max_stable_model: AbstractMaxStableModel, coordinates: AbstractCoordinates):
        temporal_obs = MaxStableAnnualMaxima.from_sampling(nb_obs, max_stable_model, coordinates)
        return cls(temporal_observations=temporal_obs, coordinates=coordinates, max_stable_model=max_stable_model)


class MarginDataset(SimulatedDataset):

    @classmethod
    def from_sampling(cls, nb_obs: int, margin_model: AbstractMarginModel, coordinates: AbstractCoordinates):
        temporal_obs = MarginAnnualMaxima.from_sampling(nb_obs, coordinates, margin_model)
        return cls(temporal_observations=temporal_obs, coordinates=coordinates, margin_model=margin_model)


class FullSimulatedDataset(SimulatedDataset):

    @classmethod
    def from_double_sampling(cls, nb_obs: int, max_stable_model: AbstractMaxStableModel,
                             coordinates: AbstractCoordinates,
                             margin_model: AbstractMarginModel):
        temporal_obs = FullAnnualMaxima.from_double_sampling(nb_obs, max_stable_model,
                                                             coordinates, margin_model)
        return cls(temporal_observations=temporal_obs, coordinates=coordinates, max_stable_model=max_stable_model,
                   margin_model=margin_model)
