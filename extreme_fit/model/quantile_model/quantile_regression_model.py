import numpy as np
from rpy2 import robjects

from extreme_fit.model.abstract_model import AbstractModel
from extreme_fit.model.result_from_model_fit.result_from_quantilreg import ResultFromQuantreg
from extreme_fit.model.utils import r, safe_run_r_estimator, get_coord_df
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractQuantileRegressionModel(AbstractModel):

    def __init__(self, dataset: AbstractDataset, quantile: float):
        self.dataset = dataset
        self.quantile = quantile

    @property
    def data(self):
        return get_coord_df(self.dataset.df_dataset)

    @property
    def first_column_of_observation(self):
        return self.data.colnames[0]

    def fit(self):
        parameters = {
            'tau': self.quantile,
            'data': self.data,
            'formula': self.formula

        }
        res = safe_run_r_estimator(r.rq, **parameters)
        return ResultFromQuantreg(res)

    @property
    def formula_str(self):
        raise NotImplementedError

    @property
    def formula(self):
        return robjects.Formula(self.first_column_of_observation + '~ ' + self.formula_str)


class ConstantQuantileRegressionModel(AbstractQuantileRegressionModel):

    @property
    def formula_str(self):
        return '1'


class TemporalCoordinatesQuantileRegressionModel(AbstractQuantileRegressionModel):

    @property
    def formula_str(self):
        assert self.dataset.coordinates.has_temporal_coordinates \
               and not self.dataset.coordinates.has_spatial_coordinates
        return AbstractCoordinates.COORDINATE_T
