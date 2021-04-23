from extreme_fit.model.quantile_model.quantile_regression_model import ConstantQuantileRegressionModel, \
    TemporalCoordinatesQuantileRegressionModel


class AbstractModelOnDailyData(object):
    pass


class ConstantQuantileRegressionModelOnDailyData(ConstantQuantileRegressionModel, AbstractModelOnDailyData):
    pass


class TemporalCoordinatesQuantileRegressionModelOnDailyData(TemporalCoordinatesQuantileRegressionModel,
                                                            AbstractModelOnDailyData):
    pass


