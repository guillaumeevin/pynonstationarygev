from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cached_property import cached_property
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_to_color
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_max_swe import CrocusSnowLoad2019
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.distribution.gumbel.gumbel_gof import goodness_of_fit_anderson, get_pvalue_anderson_darling_test
from extreme_fit.estimator.margin_estimator.utils_functions import compute_nllh_with_multiprocessing_for_large_samples
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleTemporalModel
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_trend.ensemble_fit.independent_ensemble_fit.independent_ensemble_fit import IndependentEnsembleFit
from extreme_trend.ensemble_fit.together_ensemble_fit.visualizer_non_stationary_ensemble import \
    VisualizerNonStationaryEnsemble
from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble
from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from projected_extremes.section_results.utils.combination_utils import \
    load_param_name_to_climate_coordinates_with_effects
from projected_extremes.section_results.utils.selection_utils import model_class_to_number, \
    parametrization_number_to_short_name, short_name_to_color, short_name_to_label

alpha = 0.4
linewidth_legend = 6


class VisualizerForSimpleCase(object):

    def __init__(self, altitudes, gcm_rcm_couples, safran_study_class, study_class, season, scenario,
                 model_classes: List[AbstractSpatioTemporalPolynomialModel],
                 massif_name=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False,
                 confidence_interval_based_on_delta_method=False,
                 remove_physically_implausible_models=False,
                 combinations_for_together=None,
                 weight_on_observation=1,
                 linear_effects=(False, False, False),
                 year_max_for_studies=None,
                 last_year_for_the_train_set=2019,
                 with_bootstrap_interval=True,
                 ):
        self.with_bootstrap_interval = with_bootstrap_interval
        self.linear_effects = linear_effects
        self.combinations_for_together = combinations_for_together
        self.safran_study_class = safran_study_class
        self.remove_physically_implausible_models = remove_physically_implausible_models
        self.confidence_interval_based_on_delta_method = confidence_interval_based_on_delta_method
        self.display_only_model_that_pass_gof_test = display_only_model_that_pass_gof_test
        self.model_classes = model_classes
        self.scenario = scenario
        self.season = season
        self.study_class = study_class
        self.gcm_rcm_couples = gcm_rcm_couples
        self.fit_method = fit_method
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.altitudes = altitudes
        self.massif_name = massif_name
        self.last_year_for_the_train_set = last_year_for_the_train_set

        # Load the gcm rcm couple to studies
        if year_max_for_studies is None:
            gcm_to_year_min_and_year_max = None
        else:
            gcm_to_year_min_and_year_max = {gcm: (None, year_max_for_studies) for gcm in gcm_to_color.keys()}
        gcm_rcm_couple_to_studies = VisualizerForProjectionEnsemble.load_gcm_rcm_couple_to_studies(self.altitudes,
                                                                                                   self.gcm_rcm_couples,
                                                                                                   gcm_to_year_min_and_year_max,
                                                                                                   self.safran_study_class,
                                                                                                   self.scenario,
                                                                                                   self.season,
                                                                                                   self.study_class,
                                                                                                   year_max_for_safran_study=self.last_year_for_the_train_set)

        # Add the first 50% and the last 50% of the data
        self.other_obs_visualizers = []
        studies = AltitudesStudies(safran_study_class, altitudes, season=season, year_min=1959, year_max=2019)
        visu = AltitudesStudiesVisualizerForNonStationaryModels(studies,
                                                                model_classes=[NonStationaryLocationAndScaleTemporalModel],
                                                                massif_names=[massif_name],
                                                                fit_method=fit_method,
                                                                temporal_covariate_for_fit=temporal_covariate_for_fit,
                                                                display_only_model_that_pass_anderson_test=display_only_model_that_pass_gof_test,
                                                                confidence_interval_based_on_delta_method=confidence_interval_based_on_delta_method,
                                                                remove_physically_implausible_models=remove_physically_implausible_models,
                                                                param_name_to_climate_coordinates_with_effects=None,
                                                                linear_effects=(False, False, False),
                                                                weight_on_observation=weight_on_observation)
        self.other_obs_visualizers.append(visu)
        # studies1 = AltitudesStudies(safran_study_class, altitudes, season=season, year_min=1959, year_max=1988)
        # studies2 = AltitudesStudies(safran_study_class, altitudes, season=season, year_min=1989, year_max=2019)
        # for studies in [studies1, studies2]:
        #     if issubclass(self.model_classes[0], GumbelTemporalModel):
        #         model_class_simplified = GumbelTemporalModel
        #     else:
        #         model_class_simplified = StationaryTemporalModel
        #     visu = AltitudesStudiesVisualizerForNonStationaryModels(studies,
        #                                                             model_classes=[model_class_simplified],
        #                                                             massif_names=[massif_name],
        #                                                             fit_method=fit_method,
        #                                                             temporal_covariate_for_fit=temporal_covariate_for_fit,
        #                                                             display_only_model_that_pass_anderson_test=display_only_model_that_pass_gof_test,
        #                                                             confidence_interval_based_on_delta_method=confidence_interval_based_on_delta_method,
        #                                                             remove_physically_implausible_models=remove_physically_implausible_models,
        #                                                             param_name_to_climate_coordinates_with_effects=None,
        #                                                             linear_effects=(False, False, False),
        #                                                             weight_on_observation=weight_on_observation)
        #     self.other_obs_visualizers.append(visu)

        # Load the separate fit
        try:
            self.independent_ensemble_fit = IndependentEnsembleFit([self.massif_name], gcm_rcm_couple_to_studies,
                                                                   model_classes,
                                                                   fit_method, temporal_covariate_for_fit,
                                                                   display_only_model_that_pass_gof_test,
                                                                   confidence_interval_based_on_delta_method,
                                                                   remove_physically_implausible_models,
                                                                   None)
        except AssertionError as e:
            print(e.__repr__())
            self.independent_ensemble_fit = None

        # Load the together approach without the observation
        # gcm_rcm_couple_to_studies_without_obs = {k: v for k, v in gcm_rcm_couple_to_studies.items() if k[0] != None}
        # visualizer_ensemble_without_obs = VisualizerNonStationaryEnsemble(
        #     gcm_rcm_couple_to_studies=gcm_rcm_couple_to_studies_without_obs,
        #     massif_names=[self.massif_name],
        #     model_classes=model_classes,
        #     fit_method=fit_method, temporal_covariate_for_fit=temporal_covariate_for_fit,
        #     display_only_model_that_pass_anderson_test=display_only_model_that_pass_gof_test,
        #     confidence_interval_based_on_delta_method=confidence_interval_based_on_delta_method,
        #     remove_physically_implausible_models=remove_physically_implausible_models,
        #     param_name_to_climate_coordinates_with_effects=None,
        #     linear_effects=linear_effects,
        #     weight_on_observation=weight_on_observation)

        # self.test_goodness_of_fit_obs(gcm_rcm_couple_to_studies, visualizer_ensemble_without_obs)

        # Load all the together fit approaches with observations
        # self.combination_name_to_visualizer_ensemble = {'without obs': visualizer_ensemble_without_obs}
        self.parametrization_number_to_visualizer_ensemble = {}
        if combinations_for_together is not None:
            for combination in combinations_for_together:
                param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(
                    combination)
                visualizer_ensemble = VisualizerNonStationaryEnsemble(
                    gcm_rcm_couple_to_studies=gcm_rcm_couple_to_studies,
                    massif_names=[self.massif_name],
                    model_classes=model_classes,
                    fit_method=fit_method, temporal_covariate_for_fit=temporal_covariate_for_fit,
                    display_only_model_that_pass_anderson_test=display_only_model_that_pass_gof_test,
                    confidence_interval_based_on_delta_method=confidence_interval_based_on_delta_method,
                    remove_physically_implausible_models=remove_physically_implausible_models,
                    param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
                    linear_effects=linear_effects,
                    weight_on_observation=weight_on_observation)
                self.parametrization_number_to_visualizer_ensemble[combination[0]] = visualizer_ensemble

    def visualize_gev_parameters(self):
        gev_params = GevParams.PARAM_NAMES + [True, False, None]
        for k, gev_param in enumerate(gev_params):
            # print(self.get_str(gev_param), 'plot')
            self.visualize_gev_parameter(gev_param, k)

    def get_value(self, one_fold_fit, c, gev_param):
        gev_params = one_fold_fit.best_margin_function_from_fit.get_params(c)
        if gev_param in GevParams.PARAM_NAMES:
            return gev_params.to_dict()[gev_param]
        elif gev_param is True:
            return gev_params.mean
        elif gev_param is False:
            return gev_params.std
        elif gev_param is None:
            return gev_params.return_level(50)
        else:
            raise NotImplementedError

    def get_str(self, gev_param):
        if gev_param in GevParams.PARAM_NAMES:
            return '{} parameter'.format(gev_param)
        elif gev_param is True:
            return "Mean"
        elif gev_param is False:
            return "Std"
        elif gev_param is None:
            return '50-year return level'
        else:
            raise NotImplementedError

    def visualize_gev_parameter(self, gev_param, k):
        ax = plt.gca()
        right_limit = 5
        nb_pieces_suffix = " (\#Linear pieces = {})".format(model_class_to_number[self.model_classes[0]])
        # Independent plot
        if self.independent_ensemble_fit is not None:
            items = list(self.independent_ensemble_fit.gcm_rcm_couple_to_visualizer.items())
            # Remove the safran plot with the complex model
            items = [i for i in items if i[0][0] is not None]
            # Add the safran plot with a simpler model
            for vizu in self.other_obs_visualizers:
                items.append(((None, None), vizu))

            add_label_gcm = True
            for gcm_rcm_couple, visualizer in items:
                one_fold_fit = visualizer.massif_name_to_one_fold_fit[self.massif_name]
                coordinates = one_fold_fit.best_estimator.coordinates_for_nllh
                x = [c[0] for c in coordinates if c[0] <= right_limit]
                y = [self.get_value(one_fold_fit, c, gev_param) for c in coordinates if c[0] <= right_limit]
                if gcm_rcm_couple[0] is None:
                    color = 'k'
                    label = "non-stationary GEV for the observations (\#Linear pieces = 1, constant shape)"
                    linestyle = '-'
                    linewidth = 3
                else:
                    color = 'grey'
                    linestyle = '--'
                    linewidth = 1
                    if add_label_gcm:
                        add_label_gcm = False
                        label = "non-stationary GEV for each GCM-RCM pair" + nb_pieces_suffix
                    else:
                        label = None

                ax.plot(x, y, label=label, linestyle=linestyle, color=color, linewidth=linewidth, marker=None)

        # Together plot with obs
        for j, (parametrization_number, visualizer) in enumerate(
                self.parametrization_number_to_visualizer_ensemble.items()):
            short_name = parametrization_number_to_short_name[parametrization_number]
            color = short_name_to_color[short_name]
            label = 'non-stationary GEV for the observations and all GCM-RCM pairs with\n'
            label += short_name_to_label[short_name]
            # train_score, test_score = self.combination_name_to_two_scores[combination_name]
            one_fold_fit = visualizer.massif_name_to_one_fold_fit[self.massif_name]
            coordinates = one_fold_fit.best_estimator.coordinates_for_nllh
            x = sorted([c[0] for c in coordinates if c[0] <= right_limit])
            y = [self.get_value(one_fold_fit, np.array([e]), gev_param) for e in x]
            # label = combination_name.replace('_', '-')
            # label += " ({}, {})".format(round(train_score, 2), round(test_score, 2))
            # if (k < 3) and self.linear_effects[k]:
            #     if "no effect" not in label:
            #         label += ' with linear effect'
            label += nb_pieces_suffix
            ax.plot(x, y, label=label, color=color, linewidth=3)
            # Add the slope with the added adjustment coefficients.
            # other_combinations = set([tuple(c[1:]) for c in coordinates])
            # other_combinations = [c for c in other_combinations if sum(c) > 0]
            # for last_coordinates in other_combinations:
            #     other_coordinates = [np.array([e] + list(last_coordinates)) for e in x]
            #     y = [one_fold_fit.best_margin_function_from_fit.get_params(c).to_dict()[gev_param] for c in
            #          other_coordinates]
            #     ax.plot(x, y, linestyle='--', color=color)

            # Additional plots for the value of return level
            with_significance = False
            AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
            if with_significance:
                if (gev_param is None) or (gev_param in GevParams.PARAM_NAMES):
                    # Plot the uncertainty interval
                    margin_functions = one_fold_fit.bootstrap_fitted_functions_from_fit_cached
                    coordinates_list = [np.array([t]) for t in x]

                    if gev_param is None:
                        values = [[f.get_params(c).return_level(OneFoldFit.return_period) for c in coordinates_list] for
                                  f in
                                  margin_functions]
                    else:
                        values = [[f.get_params(c).to_dict()[gev_param] for c in coordinates_list] for
                                  f in
                                  margin_functions]

                    q_list = [0.05, 0.95]
                    lower_bound, upper_bound = [np.quantile(values, q, axis=0) for q in q_list]
                    ax.fill_between(x, lower_bound, upper_bound, color=color, alpha=alpha)

        # Final plt
        ylabel = '{} for the {} massif at {} m ({})'.format(self.get_str(gev_param),
                                                            self.massif_name.replace('_', ' '),
                                                            self.altitudes[0],
                                                            visualizer.study.variable_unit)
        ylabel = ylabel[0].upper() + ylabel[1:]
        ax.set_ylabel(ylabel)
        xlabel = 'T, the smoothed anomaly of global temperature w.r.t. pre-industrial levels (K)'
        ax.set_xlabel(xlabel)
        ax.set_xlim((0, right_limit))
        handles, labels = ax.get_legend_handles_labels()
        handles[:2] = handles[:2][::-1]
        labels[:2] = labels[:2][::-1]
        ax.legend(handles, labels, prop={'size': 7})

        title = '{} massif {}'.format(self.massif_name, self.get_str(gev_param))
        visualizer.plot_name = title
        visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
        plt.close()

    def visualize_density_to_illustrate_adjustments(self, with_density=True):
        gev_param = None
        ax = plt.gca()
        ax2 = ax.twinx()

        normalization_factor = 0
        t_for_density = [1, 2, 3]

        if with_density:
            right_limit = 4
            left_limit = 0.8
            xtick = 1
        else:
            right_limit = 5.5
            left_limit = 0.5
            xtick = 0.5

        scaling_of_density = 0.75

        length_x = right_limit - left_limit

        linewidth_big_plot = 4
        ylim_upper_for_plots = 0
        if with_density:
            percentage_upper_for_plots = 0.735
        else:
            percentage_upper_for_plots = 1

        nb_pieces_suffix = " (using {} linear pieces)".format(model_class_to_number[self.model_classes[0]])
        # Independent plot
        if self.independent_ensemble_fit is not None:
            items = list(self.independent_ensemble_fit.gcm_rcm_couple_to_visualizer.items())
            # Remove the safran plot with the complex model
            items = [i for i in items if i[0][0] is not None]
            # Add the safran plot with a simpler model
            for vizu in self.other_obs_visualizers:
                items.append(((None, None), vizu))

            add_label_gcm = True
            for gcm_rcm_couple, visualizer in items:
                one_fold_fit = visualizer.massif_name_to_one_fold_fit[self.massif_name]
                coordinates = one_fold_fit.best_estimator.coordinates_for_nllh
                x = [c[0] for c in coordinates if left_limit <= c[0] <= right_limit]
                y = [self.get_value(one_fold_fit, c, gev_param) for c in coordinates if
                     left_limit <= c[0] <= right_limit]
                ylim_upper_for_plots = max(ylim_upper_for_plots, max(y))
                color, linewidth = self.setting_for_plots(gcm_rcm_couple, linewidth_big_plot)

                if gcm_rcm_couple[0] is None:
                    label = "50-year return level computed with a non-stationary GEV fitted on the observations (using 1 linear piece)"
                else:
                    if add_label_gcm:
                        add_label_gcm = False
                        label = "50-year return level computed with a non-stationary GEV fitted on each GCM-RCM pair" + nb_pieces_suffix
                    else:
                        label = None
                ax.plot(x, y, label=label, linestyle="--", color=color, linewidth=linewidth, marker=None)

            massif_name_to_ylim_upper_for_plots = {
                'Vercors': 8, 'Ubaye': 3.8
            }
            if (self.massif_name in massif_name_to_ylim_upper_for_plots) and (self.safran_study_class is CrocusSnowLoad2019):
                print('default ylim selected')
                ylim_upper_for_plots = massif_name_to_ylim_upper_for_plots[self.massif_name]

            ylim_upper = (1 / percentage_upper_for_plots) * ylim_upper_for_plots
            if with_density:
                for gcm_rcm_couple, visualizer in items:
                    one_fold_fit = visualizer.massif_name_to_one_fold_fit[self.massif_name]
                    # Only plot for the first one
                    t_list = t_for_density[:1] if gcm_rcm_couple[0] is None else t_for_density
                    for t in t_list:
                        c = np.array([t])
                        gev_params = one_fold_fit.best_margin_function_from_fit.get_params(c)
                        n = self.plot_density_vertically(ax, ax2, None, gev_params, t, ylim_upper_for_plots,
                                                         None, None, scaling_of_density)
                        normalization_factor = max(normalization_factor, n)
                for gcm_rcm_couple, visualizer in items:
                    one_fold_fit = visualizer.massif_name_to_one_fold_fit[self.massif_name]
                    color, linewidth = self.setting_for_plots(gcm_rcm_couple, linewidth_big_plot)

                    # Only plot for the first one
                    t_list = t_for_density[:1] if gcm_rcm_couple[0] is None else t_for_density
                    for t in t_list:
                        c = np.array([t])
                        gev_params = one_fold_fit.best_margin_function_from_fit.get_params(c)
                        self.plot_density_vertically(ax, ax2, color, gev_params, t, ylim_upper_for_plots,
                                                     linewidth, normalization_factor, scaling_of_density)

        # Together plot with obs
        for j, (parametrization_number, visualizer) in enumerate(
                self.parametrization_number_to_visualizer_ensemble.items()):
            short_name = parametrization_number_to_short_name[parametrization_number]
            color = short_name_to_color[short_name]
            label = 'RL50 computed with a non-stationary GEV fitted on the observations and all GCM-RCM pairs with\n'
            label += short_name_to_label[short_name]
            # train_score, test_score = self.combination_name_to_two_scores[combination_name]
            one_fold_fit = visualizer.massif_name_to_one_fold_fit[self.massif_name]
            coordinates = one_fold_fit.best_estimator.coordinates_for_nllh
            x = sorted([c[0] for c in coordinates if left_limit <= c[0] <= right_limit])
            y = [self.get_value(one_fold_fit, np.array([e]), gev_param) for e in x]
            label += nb_pieces_suffix
            ax.plot(x, y, label=label, color=color, linewidth=linewidth_big_plot,
                    linestyle="--")

            # Additional plots for the value of return level
            if self.with_bootstrap_interval:
                # Plot the uncertainty interval
                margin_functions = one_fold_fit.bootstrap_fitted_functions_from_fit_cached
                coordinates_list = [np.array([t]) for t in x]
                values = [[f.get_params(c).return_level(OneFoldFit.return_period) for c in coordinates_list] for
                          f in
                          margin_functions]
                q_list = [0.05, 0.95]
                lower_bound, upper_bound = [np.quantile(values, q, axis=0) for q in q_list]
                ax.fill_between(x, lower_bound, upper_bound, color=color, alpha=alpha)

            if with_density:
                for t in t_for_density[:]:
                    c = np.array([t])
                    scale_for_upper_arrow = 0.05
                    if j == 0:
                        text_height = 0.3
                        ax.axhline(text_height, (t + scaling_of_density - left_limit) / length_x, (t - left_limit) / length_x, color='k', linewidth=1)
                        ax.axvline(t, 0, percentage_upper_for_plots + scale_for_upper_arrow, color='k', linewidth=1,
                                   linestyle='dotted')
                        epsilon = 0.02
                        start_text = t + 0.05
                        # Draw arrow
                        size_arrow = 0.03
                        ylims = ax.get_ylim()
                        xlims = ax.get_xlim()
                        scaling_for_y = (ylims[1] - ylims[0]) / (xlims[1] - xlims[0]) / percentage_upper_for_plots
                        start_arrow = t + scaling_of_density
                        # horizontal arrow
                        ax.plot([start_arrow, start_arrow - size_arrow], [text_height, text_height + size_arrow * scaling_for_y], color='k')
                        ax.plot([start_arrow, start_arrow - size_arrow], [text_height, text_height - size_arrow * scaling_for_y], color='k')
                        ax.text(start_text, text_height + 0.1, "Probability density function")
                    gev_params = one_fold_fit.best_margin_function_from_fit.get_params(c)
                    self.plot_density_vertically(ax, ax2, color, gev_params, t, ylim_upper_for_plots,
                                                 linewidth_big_plot, normalization_factor)

        # Final plt
        ylabel = 'Annual maxima for the {} massif at {} m ({})'.format(
            self.massif_name.replace('_', ' '),
            self.altitudes[0],
            visualizer.study.variable_unit)
        ylabel = ylabel[0].upper() + ylabel[1:]
        ax.set_ylabel(ylabel)
        xlabel = 'Global warming above pre-industrial levels ($^o\\textrm{C}$)'
        ax.set_xlabel(xlabel)
        ax.set_xlim((left_limit, right_limit))
        xticks = []
        while xtick <= right_limit:
            xticks.append(xtick)
            xtick += 0.5
        ax.set_xticks(xticks)
        ax.set_xticklabels(['+{}'.format(int(h) if int(h) == h else h) for h in xticks])

        ylim = (0, ylim_upper)
        ax.set_ylim(ylim)
        ax2.set_ylim(ylim)
        ax2.set_yticks([])
        handles, labels = ax.get_legend_handles_labels()
        handles[:2] = handles[:2][::-1]
        labels[:2] = labels[:2][::-1]
        if with_density:
            legendsize = 6.4
        else:
            legendsize = 8.5
        loc = 'upper center' if with_density else "lower left"
        leg = ax.legend(handles, labels, prop={'size': legendsize}, loc=loc,
                        handlelength=4)
        # for line in leg.get_lines():
        #     line.lin
        #     line.set_linewidth(linewidth_legend)
            # line.set_linestyle("-")

        # ax.yaxis.grid()
        # self.add_second_legend(ax2, legendsize, with_density)

        title = '{} massif {}'.format(self.massif_name, self.get_str(gev_param))
        visualizer.plot_name = title
        visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
        plt.close()

    def add_second_legend(self, ax2, legendsize, with_density=True):
        # Build legend for the different style
        labels = ["Probability density function",
                  "50-year return level which equals\n"
                  "RL50 for the non-stationary\n"
                  "GEV fitted on the observations\n"
                  "and all GCM-RCM pairs",
                  "90\% uncertainty interval for\n"
                  "the 50-year return level RL50"]
        handles = [
            Line2D([0], [0], color='k', linestyle=linestyle)
            for linestyle in ["-", "--"]
        ]
        handles.append(Patch(facecolor='k', edgecolor='k', alpha=alpha))
        if not with_density:
            labels = labels[1:]
            handles = handles[1:]
        leg = ax2.legend(handles=handles, labels=labels, loc='upper right', prop={'size': legendsize},
                         handlelength=3)
        for line in leg.get_lines():
            line.set_linewidth(linewidth_legend)

    def setting_for_plots(self, gcm_rcm_couple, linewidth_big_plot):
        if gcm_rcm_couple[0] is None:
            color = 'dimgrey'
            linewidth = linewidth_big_plot


        else:
            color = 'lightgrey'
            linewidth = 1
        return color, linewidth

    def plot_density_vertically(self, ax, ax2, color, gev_params: GevParams, t, ylim_upper_for_plots, linewidth,
                                normalize_factor=None, scaling_of_density=1):
        real_density = True
        if real_density:
            yp = np.linspace(0, ylim_upper_for_plots, num=1000)
            xp = gev_params.density(yp)
        else:
            sns.kdeplot(gev_params.sample(10000), ax=ax2, label=None, color=color,
                        vertical=True, linewidth=0)
            xp, yp = ax2.get_lines()[-1].get_data()
        if normalize_factor is None:
            return xp.max()
        else:
            xp = xp * scaling_of_density / normalize_factor
            # xp = t + -xp
            xp = t + xp
            l = [(xpp, ypp) for xpp, ypp in zip(xp, yp) if ypp <= ylim_upper_for_plots]
            xp, yp = zip(*l)
            ax.plot(xp, yp, color=color, linestyle='-', linewidth=linewidth)

    def test_goodness_of_fit_obs(self, gcm_rcm_couple_to_studies, visualizer):
        margin_function = visualizer.massif_name_to_one_fold_fit[self.massif_name].best_margin_function_from_fit
        # Test if the observation can be explained by the model
        obs_dataset = gcm_rcm_couple_to_studies[(None, None)].spatio_temporal_dataset(self.massif_name)
        coordinates_values = obs_dataset.coordinates.df_temporal_coordinates_for_fit(
            temporal_covariate_for_fit=self.temporal_covariate_for_fit,
            drop_duplicates=False,
            for_fit=True).values
        gumbel_quantiles = []
        for maximum, coordinates in zip(obs_dataset.maxima_gev, coordinates_values):
            gev_params = margin_function.get_params(coordinates)
            maximum_standardized = gev_params.gumbel_standardization(maximum[0])
            gumbel_quantiles.append(maximum_standardized)
        test_result = goodness_of_fit_anderson(gumbel_quantiles)
        print('Test result:', test_result, get_pvalue_anderson_darling_test(gumbel_quantiles))

    @cached_property
    def combination_name_to_two_scores(self):
        combination_name_to_two_scores = {}
        for combination_name, visualizer_together in self.parametrization_number_to_visualizer_ensemble.items():
            best_estimator = visualizer_together.massif_name_to_one_fold_fit[self.massif_name].best_estimator

            studies_train = AltitudesStudies(self.safran_study_class, self.altitudes, season=self.season,
                                             year_max=self.last_year_for_the_train_set)

            studies_test = AltitudesStudies(self.safran_study_class, self.altitudes, season=self.season,
                                            year_min=self.last_year_for_the_train_set + 1)

            two_scores = []
            for studies in [studies_train, studies_test][:]:
                dataset_test = studies.spatio_temporal_dataset(self.massif_name)
                df_coordinates_temp_for_test = best_estimator.load_coordinates_temp(dataset_test.coordinates,
                                                                                    for_fit=False)
                maxima_values = dataset_test.observations.maxima_gev
                coordinate_values = df_coordinates_temp_for_test.values
                nllh = compute_nllh_with_multiprocessing_for_large_samples(coordinate_values, maxima_values,
                                                                           best_estimator.margin_function_from_fit,
                                                                           True, True, False)
                score = nllh / len(coordinate_values)
                print(combination_name, score)
                two_scores.append(score)
            combination_name_to_two_scores[combination_name] = tuple(two_scores)
        return combination_name_to_two_scores
