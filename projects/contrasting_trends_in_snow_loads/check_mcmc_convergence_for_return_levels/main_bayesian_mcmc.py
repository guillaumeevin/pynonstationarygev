import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from extreme_data.meteo_france_data.scm_models_data.visualization.utils import create_adjusted_axes
from extreme_data.exceeding_snow_loads.check_mcmc_convergence_for_return_levels.gelman_convergence_test import \
    compute_gelman_score
from extreme_fit.estimator.margin_estimator.utils import fitted_linear_margin_estimator
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    TemporalMarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    ExtractEurocodeReturnLevelFromMyBayesianExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from spatio_temporal_dataset.utils import load_temporal_coordinates_and_dataset


def main_drawing_bayesian(N=10000):
    nb_chains = 3
    means, variances = [], []
    for i in range(nb_chains):
        result_from_fit = plot_chain(N, show=False).result_from_fit
        means.append(result_from_fit.mean_posterior_parameters)
        variances.append(result_from_fit.variance_posterior_parameters)
    means, variances = pd.DataFrame(means).transpose(), pd.DataFrame(variances).transpose()
    scores = []
    for (_, row1), (_, row2) in zip(means.iterrows(), variances.iterrows()):
        score = compute_gelman_score(row1, row2, N, nb_chains)
        scores.append(score)
    print(scores)


def plot_chain(N=10000, show=True):
    return_level_bayesian = get_return_level_bayesian_example(N * 2)
    print(return_level_bayesian.result_from_fit.df_all_samples)
    print(return_level_bayesian.result_from_fit.df_posterior_samples)
    print(return_level_bayesian.posterior_eurocode_return_level_samples_for_temporal_covariate)
    axes = create_adjusted_axes(1, 3)
    # Plot the trajectories on the first axis
    ax_trajectories = axes[0]
    ax_trajectories_inverted = ax_trajectories.twinx()
    df_all_samples = return_level_bayesian.result_from_fit.df_all_samples
    iteration_step = [i + 1 for i in df_all_samples.index]
    lns = []
    # Last color is for the return level
    colors = ['r', 'g', 'b', 'tab:orange']
    gev_param_name_to_color = dict(zip(GevParams.PARAM_NAMES, colors))
    gev_param_name_to_ax = dict(
        zip(GevParams.PARAM_NAMES, [ax_trajectories, ax_trajectories, ax_trajectories]))
    # zip(GevParams.PARAM_NAMES, [ax_trajectories, ax_trajectories, ax_trajectories_inverted]))
    gev_param_name_to_label = {n: GevParams.greek_letter_from_gev_param_name(n) for n in GevParams.PARAM_NAMES}
    for j, gev_param_name in enumerate(GevParams.PARAM_NAMES[:]):
        label = gev_param_name_to_label[gev_param_name]
        ax = gev_param_name_to_ax[gev_param_name]
        color = gev_param_name_to_color[gev_param_name]
        ln = ax.plot(iteration_step, df_all_samples.iloc[:, j].values, label=label, color=color)
        lns.extend(ln)
    ax_trajectories_inverted.set_ylim(-0.3, 10)
    ax_trajectories.set_ylim(-0.3, 90)
    for ax in [ax_trajectories, ax_trajectories_inverted]:
        ax.set_xlim(min(iteration_step), max(iteration_step))
    # labs = [l.get_label() for l in lns]
    # ax_trajectories.legend(lns, labs, loc=0)
    ax.axvline(x=return_level_bayesian.result_from_fit.burn_in_nb, color='k', linestyle='--')
    ax_trajectories_inverted.legend(loc=1)
    ax_trajectories.legend(loc=2)
    ax_trajectories.set_xlabel("MCMC iterations")
    # Plot the parameter posterior on axes 1
    ax_parameter_posterior = axes[1]
    ax_parameter_posterior_flip = ax_parameter_posterior.twiny()
    gev_param_name_to_ax = dict(
        zip(GevParams.PARAM_NAMES,
            [ax_parameter_posterior, ax_parameter_posterior.twiny(), ax_parameter_posterior_flip]))
    df_posterior_samples = return_level_bayesian.result_from_fit.df_posterior_samples
    lns = []
    for j, gev_param_name in enumerate(GevParams.PARAM_NAMES[:]):
        label = gev_param_name_to_label[gev_param_name]
        color = gev_param_name_to_color[gev_param_name]
        ax = gev_param_name_to_ax[gev_param_name]
        ln = sns.kdeplot(df_posterior_samples.iloc[:, j], ax=ax, label=label, color=color)
        lns.append(ln)
    labs = [l.get_label() for l in lns]
    ax_parameter_posterior.legend(lns, labs, loc=0)
    # Plot the return level posterior on the last axes
    ax_return_level_posterior = axes[2]
    sns.kdeplot(return_level_bayesian.posterior_eurocode_return_level_samples_for_temporal_covariate,
                ax=ax_return_level_posterior, color=colors[-1])
    ax_return_level_posterior.set_xlabel("$z_p(\\theta)$")
    ax_return_level_posterior.set_ylabel("$p(z_p(\\theta)|y)$")
    if show:
        plt.show()
    return return_level_bayesian


def get_return_level_bayesian_example(nb_iterations_for_bayesian_fit):
    # It converges well because we also take the zero values into account
    maxima, years = CrocusSnowLoadTotal(altitude=1800).annual_maxima_and_years('Vercors')
    # plt.plot(years, maxima)
    # plt.show()
    model_class = StationaryTemporalModel
    coordinates, dataset = load_temporal_coordinates_and_dataset(maxima, years)
    fitted_estimator = fitted_linear_margin_estimator(model_class, coordinates, dataset, starting_year=1959,
                                                      fit_method=TemporalMarginFitMethod.extremes_fevd_bayesian,
                                                      nb_iterations_for_bayesian_fit=nb_iterations_for_bayesian_fit)
    return_level_bayesian = ExtractEurocodeReturnLevelFromMyBayesianExtremes(estimator=fitted_estimator,
                                                                             ci_method=ConfidenceIntervalMethodFromExtremes.my_bayes,
                                                                             temporal_covariate=2019)
    return return_level_bayesian


if __name__ == '__main__':
    main_drawing_bayesian()
    plt.plot()
    # plot_chain()
