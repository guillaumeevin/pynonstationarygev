from extreme_estimator.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_estimator.extreme_models.margin_model.linear_margin_model import LinearAllParametersAllDimsMarginModel, \
    ConstantMarginModel
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class LinearMarginEstimator_LinearAllParametersAllDims(LinearMarginEstimator):

    @classmethod
    def from_dataset(cls, dataset: AbstractDataset):
        return cls(dataset, LinearAllParametersAllDimsMarginModel(dataset.coordinates))


class LinearMarginEstimator_Constant(LinearMarginEstimator):

    @classmethod
    def from_dataset(cls, dataset: AbstractDataset):
        return cls(dataset, ConstantMarginModel(dataset.coordinates))


MARGIN_ESTIMATORS_FOR_SIMULATION = [
    LinearMarginEstimator_LinearAllParametersAllDims,
    LinearMarginEstimator_Constant,
]
