from extreme_fit.estimator.full_estimator.abstract_full_estimator import \
    FullEstimatorInASingleStepWithSmoothMargin
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearAllParametersAllDimsMarginModel, \
    ConstantMarginModel
from extreme_fit.model.max_stable_model.max_stable_models import Smith
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class FullEstimatorInASingleStepWithSmoothMargin_LinearAllParametersAllDims_Smith(
    FullEstimatorInASingleStepWithSmoothMargin):

    @classmethod
    def from_dataset(cls, dataset: AbstractDataset):
        return cls(dataset, margin_model=LinearAllParametersAllDimsMarginModel(dataset.coordinates),
                   max_stable_model=Smith())


class FullEstimatorInASingleStepWithSmoothMargin_Constant_Smith(FullEstimatorInASingleStepWithSmoothMargin):

    @classmethod
    def from_dataset(cls, dataset: AbstractDataset):
        return cls(dataset, margin_model=ConstantMarginModel(dataset.coordinates),
                   max_stable_model=Smith())


FULL_ESTIMATORS_FOR_SIMULATION = [
    FullEstimatorInASingleStepWithSmoothMargin_LinearAllParametersAllDims_Smith,
    FullEstimatorInASingleStepWithSmoothMargin_Constant_Smith,

]
