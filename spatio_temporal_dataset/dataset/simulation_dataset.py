from extreme_estimator.R_fit.max_stable_fit.max_stable_models import MaxStableModel, GaussianMSP
import pandas as pd
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.temporal_maxima.temporal_maxima import TemporalMaxima
from spatio_temporal_dataset.spatial_coordinates.abstract_coordinates import AbstractSpatialCoordinates
from spatio_temporal_dataset.spatial_coordinates.generated_coordinates import CircleCoordinates


class SimulatedDataset(AbstractDataset):

    def __init__(self, temporal_maxima: TemporalMaxima, spatial_coordinates: AbstractSpatialCoordinates,
                 max_stable_model: MaxStableModel):
        super().__init__(temporal_maxima, spatial_coordinates)
        self.max_stable_model = max_stable_model

    @classmethod
    def from_max_stable_sampling(cls, nb_obs: int, max_stable_model:MaxStableModel, spatial_coordinates: AbstractSpatialCoordinates):
        maxima = max_stable_model.rmaxstab(nb_obs=nb_obs, coord=spatial_coordinates.coord)
        df_maxima = pd.DataFrame(data=maxima, index=spatial_coordinates.index)
        temporal_maxima = TemporalMaxima(df_maxima=df_maxima)
        return cls(temporal_maxima=temporal_maxima, spatial_coordinates=spatial_coordinates, max_stable_model=max_stable_model)


if __name__ == '__main__':
    coord = CircleCoordinates.from_nb_points(nb_points=5, max_radius=1)
    max_stable_model = GaussianMSP()
    dataset = SimulatedDataset.from_max_stable_sampling(nb_obs=50, max_stable_model=max_stable_model, spatial_coordinates=coord)
    print(dataset.df_dataset)




