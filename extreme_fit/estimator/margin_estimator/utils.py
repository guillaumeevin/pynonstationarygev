from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator


def fitted_linear_margin_estimator(model_class, coordinates, dataset, starting_year, fit_method, **model_kwargs):
    model = model_class(coordinates, starting_point=starting_year, fit_method=fit_method, **model_kwargs)
    estimator = LinearMarginEstimator(dataset, model)
    estimator.fit()
    return estimator
