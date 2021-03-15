import numpy as np
import pandas as pd
from extreme_fit.estimator.margin_estimator.utils import fitted_linear_margin_estimator
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from extreme_fit.model.result_from_model_fit.result_from_extremes.result_from_bayesian_extremes import \
    ResultFromBayesianExtremes
from extreme_fit.model.utils import r
from spatio_temporal_dataset.utils import load_temporal_coordinates_and_dataset


def compute_gelman_score(means, variances, N, M):
    assert isinstance(means, pd.Series)
    assert isinstance(variances, pd.Series)
    mean = means.mean()
    B = N * (means - mean).pow(2).sum_of_differences() / (M - 1)
    W = variances.mean()
    V_hat = (N - 1) * W / N
    V_hat += (M + 1) * B / (M * N)
    return V_hat / W


def compute_refined_gelman_score(means, variances, N, M):
    R = compute_gelman_score(means, variances, N, M)
    # todo: check how to d
    # d = 2 * V_hat / W
    # R = (d + 3) * V_hat
    # R /= (d + 1) * W
    # return np.sqrt(R)


def compute_gelman_convergence_value(non_null_years_and_maxima, mcmc_iterations, model_class, nb_chains):
    s_means, s_variances = [], []
    df_params_initial_fit_bayesian = sample_starting_value(nb_chains)
    for i, row in df_params_initial_fit_bayesian.iterrows():
        s_mean, s_variance = compute_mean_and_variance(mcmc_iterations, model_class, non_null_years_and_maxima,
                                                       params_initial_fit_bayesian=row.to_dict())
        s_means.append(s_mean)
        s_variances.append(s_variance)
    df_mean = pd.concat(s_means, axis=1).transpose()
    df_variance = pd.concat(s_variances, axis=1).transpose()
    param_name_to_R = {param_name: compute_gelman_score(df_mean[param_name], df_variance[param_name], N=mcmc_iterations // 2, M=nb_chains)
                       for param_name in df_params_initial_fit_bayesian.columns}
    print(param_name_to_R)
    return max(param_name_to_R.values())


def sample_starting_value(nb_chains):
    return pd.DataFrame({
        'shape': np.array(r.rbeta(nb_chains, 6, 9)) - 0.5,
        'location': np.array(r.rnorm(nb_chains, 0, 1)),
        'scale': np.array(r.rexp(nb_chains, 1)),
    })


def compute_mean_and_variance(mcmc_iterations, model_class, non_null_years_and_maxima, params_initial_fit_bayesian):
    maxima, years = non_null_years_and_maxima
    coordinates, dataset = load_temporal_coordinates_and_dataset(maxima, years)
    fitted_estimator = fitted_linear_margin_estimator(model_class, coordinates, dataset, starting_year=None,
                                                      fit_method=MarginFitMethod.extremes_fevd_bayesian,
                                                      nb_iterations_for_bayesian_fit=mcmc_iterations,
                                                      params_initial_fit_bayesian=params_initial_fit_bayesian)
    res = fitted_estimator.result_from_model_fit  # type: ResultFromBayesianExtremes
    return res.mean_posterior_parameters, res.variance_posterior_parameters


#
def test_gelman():
    M = 3
    epsi = 1e-2
    means = pd.Series([1 + i *epsi for i in range(M)])
    variances = pd.Series([1 for _ in range(M)])
    N = 5000
    res = compute_gelman_score(means, variances, M, N)
    print(res)


if __name__ == '__main__':
    test_gelman()
