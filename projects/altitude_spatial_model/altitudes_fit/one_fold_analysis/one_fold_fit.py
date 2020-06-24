from extreme_fit.estimator.margin_estimator.utils import fitted_linear_margin_estimator_short
from extreme_fit.model.margin_model.utils import MarginFitMethod


class OneFoldFit(object):

    def __init__(self, dataset, models_classes, fit_method=MarginFitMethod.extremes_fevd_mle):
        self.dataset = dataset
        self.models_classes = models_classes
        self.fit_method = fit_method

        # Fit Estimators
        self.model_class_to_estimator = {}
        for model_class in models_classes:
            self.model_class_to_estimator[model_class] = fitted_linear_margin_estimator_short(model_class=model_class,
                                                                                              dataset=self.dataset,
                                                                                              fit_method=self.fit_method)
        # Some display
        for estimator in self.model_class_to_estimator.values():
            print(estimator.result_from_model_fit.aic)
