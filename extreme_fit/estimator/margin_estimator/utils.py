from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator


def fitted_linear_margin_estimator_short(model_class, dataset, fit_method, drop_duplicates=None,
                                         param_name_to_climate_coordinates_with_effects=None,
                                         linear_effects=(False, False, False),
                                         **model_kwargs) -> LinearMarginEstimator:
    return fitted_linear_margin_estimator(model_class, dataset.coordinates, dataset, None,
                                          fit_method, drop_duplicates, param_name_to_climate_coordinates_with_effects,
                                          linear_effects,
                                          **model_kwargs)


def fitted_linear_margin_estimator(model_class, coordinates, dataset, starting_year, fit_method, drop_duplicates=None,
                                   param_name_to_climate_coordinates_with_effects=None, linear_effects=(False, False, False),
                                   **model_kwargs):
    model = model_class(coordinates, starting_point=starting_year,
                        fit_method=fit_method, param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
                        linear_effects=linear_effects,
                        **model_kwargs)
    if drop_duplicates is not None:
        model.drop_duplicates = drop_duplicates
    estimator = LinearMarginEstimator(dataset, model)
    estimator.fit()
    return estimator
