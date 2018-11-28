import numpy as np
import pandas as pd

from extreme_estimator.extreme_models.margin_model.smooth_margin_model import LinearMarginModel, \
    LinearAllParametersAllDimsMarginModel
from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import CovarianceFunction
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Schlather
from extreme_estimator.extreme_models.utils import r, set_seed_r
from extreme_estimator.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.dataset.simulation_dataset import FullSimulatedDataset
from spatio_temporal_dataset.temporal_observations.annual_maxima_observations import MaxStableAnnualMaxima

print('R')
set_seed_r()
r("""
n.site <- 30
locations <- matrix(runif(2*n.site, 0, 10), ncol = 2)
colnames(locations) <- c("lon", "lat")""")

for _ in range(2):
    set_seed_r()
    r("""data <- rmaxstab(40, locations, cov.mod = "whitmat", nugget = 0, range = 3,
    smooth = 0.5)""")
    print(np.sum(r.data))
    r("""
    ##param.loc <- -10 + 2 * locations[,2]
    ##TODO-IMPLEMENT SQUARE: param.scale <- 5 + 2 * locations[,1] + locations[,2]^2
    ##param.scale <- 5 + 2 * locations[,1] + locations[,2]
    param.shape <- rep(0.2, n.site)
    param.loc <- rep(0.2, n.site)
    param.scale <- rep(0.2, n.site)
    ##Transform the unit Frechet margins to GEV
    for (i in 1:n.site)
    data[,i] <- frech2gev(data[,i], param.loc[i], param.scale[i], param.shape[i])""")
    print(np.sum(r.data))

print('\n\nPython')

params_sample = {'range': 3, 'smooth': 0.5, 'nugget': 0.0}
max_stable_model = Schlather(covariance_function=CovarianceFunction.whitmat, params_sample=params_sample)
df = pd.DataFrame(data=r.locations, columns=AbstractCoordinates.COORDINATE_NAMES[:2])
coordinates = AbstractCoordinates.from_df(df)
set_seed_r()
maxima = MaxStableAnnualMaxima.from_sampling(nb_obs=40, max_stable_model=max_stable_model, coordinates=coordinates)
print(np.sum(maxima.maxima_frech))

# gev_param_name_to_coef_list = {
#     GevParams.GEV_LOC: [-10, 0, 2],
#     GevParams.GEV_SHAPE: [5, 2, 1],
#     GevParams.GEV_SCALE: [0.2, 0, 0],
# }
gev_param_name_to_coef_list = {
    GevParams.GEV_LOC: [0.2, 0, 0],
    GevParams.GEV_SHAPE: [0.2, 0, 0],
    GevParams.GEV_SCALE: [0.2, 0, 0],
}
margin_model = LinearAllParametersAllDimsMarginModel.from_coef_list(coordinates, gev_param_name_to_coef_list)
maxima_gev = margin_model.frech2gev(maxima.maxima_frech, coordinates.coordinates_values, margin_model.margin_function_sample)
print(np.sum(maxima_gev))

# dataset = FullSimulatedDataset.from_double_sampling(nb_obs=nb_obs, margin_model=margin_model,
#                                                     coordinates=coordinates,
#                                                     max_stable_model=max_stable_model)

