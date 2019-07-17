import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.abstract_hypercube_visualizer import \
    AbstractHypercubeVisualizer
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    SCM_STUDY_NAME_TO_COLOR, SCM_STUDY_NAME_TO_ABBREVIATION, SCM_STUDY_CLASS_TO_ABBREVIATION, SCM_STUDIES_NAMES
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
from experiment.trend_analysis.univariate_test.abstract_gev_trend_test import AbstractGevTrendTest
from experiment.trend_analysis.univariate_test.abstract_univariate_test import AbstractUnivariateTest
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from utils import get_display_name_from_object_type

ALTITUDES_XLABEL = 'altitudes'

STARTING_YEARS_XLABEL = 'starting years'

from math import log10, floor


def round_sig(x, sig=2):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


class AltitudeHypercubeVisualizer(AbstractHypercubeVisualizer):

    @property
    def altitudes(self):
        return self.tuple_values(idx=0)

    @property
    def display_trend_type_to_style(self):
        return self.trend_test_class.display_trend_type_to_style()

    @property
    def display_trend_types(self):
        return self.display_trend_type_to_style.keys()

    @property
    def nb_rows(self):
        return 1

    def ylabel_to_series(self, reduction_function, isin_parameters=None):
        return {}

    def trend_type_to_series(self, reduction_function, isin_parameters=None):
        # Map each trend type to its serie with percentages
        # Define here all the trend type we might need in the results/displays
        return {trend_type: self.trend_type_reduction_series(reduction_function=reduction_function,
                                                             df_bool=self.df_bool(trend_type, isin_parameters).copy())
                for trend_type in self.trend_types_to_process}

    @property
    def trend_types_to_process(self):
        return list(self.display_trend_types) + [AbstractUnivariateTest.SIGNIFICATIVE_ALL_TREND]

    def df_bool(self, display_trend_type, isin_parameters=None):
        return self.df_hypercube_trend_type.isin(AbstractUnivariateTest.get_real_trend_types(display_trend_type))

    def trend_type_reduction_series(self, reduction_function, df_bool):
        # Reduce df_bool df to a serie s_trend_type_percentage
        s_trend_type_percentage = reduction_function(df_bool)
        assert isinstance(s_trend_type_percentage, pd.Series)
        assert not isinstance(s_trend_type_percentage.index, pd.MultiIndex)
        s_trend_type_percentage *= 100
        series = [s_trend_type_percentage]
        if self.reduce_strength_array:
            # Reduce df_strength to a serie s_trend_strength
            df_strength = self.df_hypercube_trend_slope_relative_strength[df_bool]
            s_trend_strength = reduction_function(df_strength)
            df_constant = self.df_hypercube_trend_constant_quantile[df_bool]
            s_trend_constant = reduction_function(df_constant)
            series.extend([s_trend_strength, s_trend_constant])
        return series

    def subtitle_to_reduction_function(self, reduction_function, level=None, add_detailed_plot=False, subtitle=None):
        def reduction_function_with_level(df_bool, **kwargs):
            return reduction_function(df_bool, **kwargs) if level is None else reduction_function(df_bool, level,
                                                                                                  **kwargs)

        if subtitle is None:
            # subtitle = self.study.variable_name[:6]
            subtitle = SCM_STUDY_CLASS_TO_ABBREVIATION[type(self.study)]
            # Ensure that subtitle does not belong to this dictionary so that the plot will be normal
            assert subtitle not in SCM_STUDY_NAME_TO_COLOR

        return {subtitle: reduction_function_with_level}

    def get_title_plot(self, xlabel, ax_idx=None):
        labels = ['altitudes', 'starting years', 'massifs']
        assert xlabel in labels, xlabel
        if ax_idx == 1:
            return '% of change per year for the parameter value'
        elif ax_idx == 0:
            return '% of trend type'
        else:
            labels.remove(xlabel)
            if xlabel != 'starting years':
                labels.remove('starting years')
            common_txt = 'averaged on {}'.format(' & '.join(labels))
            common_txt += ' with any starting year between {} and {}'.format(self.first_starting_year,
                                                                             self.last_starting_year)
            return common_txt

    def visualize_trend_test_evolution(self, reduction_function, xlabel, xlabel_values, axes=None, marker='o',
                                       subtitle='', isin_parameters=None,
                                       plot_title=None, idx_reduction=None,
                                       poster_plot=False):

        # Plot in one graph several graph that correspond to the same trend_type
        trend_type_to_series = self.trend_type_to_series(reduction_function, isin_parameters)
        end_idx = len(list(trend_type_to_series.values())[0])
        axes_for_trend_type = axes[:end_idx]
        for ax_idx, ax in enumerate(axes_for_trend_type):
            for display_trend_type in self.display_trend_types:
                style = self.display_trend_type_to_style[display_trend_type]
                values = trend_type_to_series[display_trend_type][ax_idx]
                xlabel_values = list(values.index)
                values = list(values.values)
                ax.plot(xlabel_values, values, style + marker, label=display_trend_type)

            if ax_idx == 0:
                # Global information
                ax.set_ylabel(self.get_title_plot(xlabel, ax_idx=0))
                if xlabel != STARTING_YEARS_XLABEL:
                    ax.set_yticks(list(range(0, 101, 20)))
            else:
                ax.set_ylabel(self.get_title_plot(xlabel, ax_idx=ax_idx))

            # Common function functions
            if xlabel == STARTING_YEARS_XLABEL:
                ax.set_xticks(xlabel_values[::3])
            else:
                ax.set_xticks(xlabel_values)
            ax.set_xlabel(xlabel)
            ax.grid()
            ax.legend()
            if plot_title is not None:
                ax.set_title(plot_title)

        # Plot other graphs where there is a single line that do not correspond to trend types
        axes_remaining = axes[end_idx:]
        ylabel_to_series = self.ylabel_to_series(reduction_function, isin_parameters)
        assert len(axes_remaining) == len(ylabel_to_series), '{}, {}'.format(len(axes_remaining), len(ylabel_to_series))
        best_year = np.nan
        for ax_idx, (ax, (ylabel, serie)) in enumerate(zip(axes_remaining, ylabel_to_series.items())):
            assert isinstance(serie, pd.Series)
            xlabel_values = list(serie.index)
            values = list(serie.values)
            argmax_idx = np.argmax(values)
            best_year = xlabel_values[argmax_idx]
            if plot_title is not None:
                plot_title += ' (max reached in {})'.format(best_year)

            if subtitle in SCM_STUDY_NAME_TO_COLOR:
                ax_reversed, color = ax.twinx(), SCM_STUDY_NAME_TO_COLOR[subtitle]
                ylabel = 'mean logLik for ' + SCM_STUDY_NAME_TO_ABBREVIATION[subtitle]
                ax.plot([], [], label=ylabel, color=color)
                linewidth = 10 if poster_plot else None
                ax_reversed.plot(xlabel_values, values, label=ylabel, color=color, linewidth=linewidth)
                fontsize = 30 if poster_plot else None
                ax_reversed.set_ylabel(ylabel, color=color, fontsize=fontsize, labelpad=-20)
                ax_reversed.axvline(x=best_year, color=color, linestyle='--', linewidth=linewidth)

                # Offset the right spine of par2.  The ticks and label have already been
                # placed on the right by twinx above.
                position = 1 + idx_reduction * 0.08
                if idx_reduction > 0:
                    ax_reversed.spines["right"].set_position(("axes", position))
                    # Having been created by twinx, par2 has its frame off, so the line of its
                    # detached spine is invisible.  First, activate the frame but make the patch
                    # and spines invisible.
                    make_patch_spines_invisible(ax_reversed)
                    # Second, show the right spine.
                    ax_reversed.spines["right"].set_visible(True)
                if poster_plot:
                    # ax_reversed.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax_reversed.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                    # ax_reversed.tick_params(axis='both', which='major', labelsize=15)
                    ax_reversed.tick_params(axis='y', which='major', labelsize=25)
                    # ax.tick_params(axis='x', which='major', labelsize=20)
                    ax.tick_params(axis='x', which='major', labelsize=25)

                    ax_reversed.yaxis.set_ticks([np.round(min(values), 1), np.round(max(values), 1)])
            else:
                ax.set_title(plot_title)
                # ax.legend()
            # Common things to all the graph
            if not poster_plot:
                ax.set_xlabel(xlabel)
            plt.setp(ax.get_yticklabels(), visible=False)

        specific_title = self.specific_title_trend_evolution(subtitle, xlabel, loglik_title=len(axes_remaining) > 0)

        # Figure title
        # specific_title += '\n'
        #
        # trend_types = [AbstractUnivariateTest.ALL_TREND,
        #                AbstractUnivariateTest.SIGNIFICATIVE_ALL_TREND,
        #                AbstractUnivariateTest.SIGNIFICATIVE_POSITIVE_TREND,
        #                AbstractUnivariateTest.SIGNIFICATIVE_NEGATIVE_TREND]
        # series = [trend_type_to_series[trend_type][0] for trend_type in trend_types]
        # percents = [serie.sum() if xlabel == STARTING_YEARS_XLABEL else serie.mean() for serie in series]
        # percents = [np.round(p) for p in percents]
        # specific_title += 'Total ' if xlabel == STARTING_YEARS_XLABEL else 'Mean '
        # specific_title += 'all trend {}, all significative trends: {} (+:{}  -{})'.format(*percents)
        if not poster_plot:
            plt.suptitle(specific_title)

        return specific_title, best_year

    def specific_title_trend_evolution(self, subtitle, xlabel, loglik_title=False):
        if loglik_title:
            specific_title = 'Mean LogLik of the non stationary model'
        else:
            specific_title = 'Evolution of {} trends'.format(subtitle)
        specific_title += ' wrt to the {}'.format(xlabel)
        if len(self.altitudes) == 1:
            specific_title += ' at altitude={}m'.format(self.altitudes[0])
        return specific_title

    def load_trend_test_evolution_axes(self, nb_rows):
        fig, axes = plt.subplots(nb_rows, 1, figsize=self.study_visualizer.figsize, constrained_layout=True)
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        return axes

    def load_trend_test_evolution_axes_with_columns(self, nb_rows, nb_columns):
        fig, axes = plt.subplots(nb_rows, nb_columns, figsize=self.study_visualizer.figsize, constrained_layout=True)
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        else:
            axes = axes.reshape((nb_rows * nb_columns))
        return axes

    def visualize_trend_test_repartition(self, reduction_function, axes=None, subtitle='', isin_parameters=None,
                                         plot_title=None):

        for i, axes_row in enumerate(axes):
            trend_type_to_serie = {k: v[i].replace(0.0, np.nan) for k, v in
                                   self.trend_type_to_series(reduction_function, isin_parameters).items()}
            vmax = max([s.max() for s in trend_type_to_serie.values()])
            vmin = min([s.min() for s in trend_type_to_serie.values()])
            vmax = max(vmax, 0.01)
            if vmin == vmax:
                epislon = 0.001 * vmax
                vmin -= epislon
                vmax += epislon

            if i == 0:
                vmin, vmax = 0, 100
            for ax, display_trend_type in zip(axes_row, self.display_trend_types):
                serie = trend_type_to_serie[display_trend_type]
                massif_to_value = dict(serie)
                cmap = self.trend_test_class.get_cmap_from_trend_type(display_trend_type)
                self.study.visualize_study(ax, massif_to_value, show=False, cmap=cmap, label=display_trend_type,
                                           vmax=vmax, vmin=vmin)
                if plot_title is not None:
                    ax.set_title(plot_title)
            row_title = self.get_title_plot(xlabel='massifs', ax_idx=i)
            StudyVisualizer.clean_axes_write_title_on_the_left(axes_row, row_title, left_border=None)

        # Global information
        title = 'Repartition of {} trends (significative or not) with {}'.format(subtitle, self.trend_test_name)
        title += '\n ' + self.get_title_plot('massifs')
        plt.suptitle(title)

        return title

    def visualize_trend_test_repartition_poster(self, reduction_function, axes=None, subtitle='', isin_parameters=None,
                                                plot_title=None):
        trend_type_to_serie = {k: v[0].replace(0.0, np.nan) for k, v in
                               self.trend_type_to_series(reduction_function, isin_parameters).items()}

        massif_to_color = {}
        add_text = self.nb_rows > 1
        massif_to_year = {}
        massif_to_strength = {}
        massif_to_constant = {}
        poster_trend_types = [AbstractUnivariateTest.SIGNIFICATIVE_POSITIVE_TREND,
                              AbstractUnivariateTest.SIGNIFICATIVE_NEGATIVE_TREND,
                              AbstractUnivariateTest.NEGATIVE_TREND,
                              AbstractUnivariateTest.POSITIVE_TREND,
                              ][:]
        for display_trend_type, style in self.display_trend_type_to_style.items():
            if display_trend_type in poster_trend_types:
                color = style[:-1]
                serie = trend_type_to_serie[display_trend_type]
                massif_to_color_for_trend_type = {k: color for k, v in dict(serie).items() if not np.isnan(v)}
                massif_to_color.update(massif_to_color_for_trend_type)
                if add_text:
                    if self.reduce_strength_array:
                        massif_to_value_for_trend_type = [{k: v for k, v in
                                                           self.trend_type_to_series(reduction_function,
                                                                                     isin_parameters)[
                                                               display_trend_type][i].items()
                                                           if k in massif_to_color_for_trend_type} for i in [1, 2]]
                        massif_to_strength.update(massif_to_value_for_trend_type[0])
                        massif_to_constant.update(massif_to_value_for_trend_type[1])
                    else:
                        massif_to_value_for_trend_type = {k: int(v) for k, v in
                                                          self.trend_type_to_series(reduction_function,
                                                                                    isin_parameters)[
                                                              display_trend_type][1].items()
                                                          if k in massif_to_color_for_trend_type}
                        massif_to_year.update(massif_to_value_for_trend_type)
        # Compute massif_to_value
        if self.reduce_strength_array:
            massif_name_to_value = {m: "{} {}{}".format(
                                                                      int(massif_to_constant[m]),
                                                                      "+" if massif_to_strength[m] > 0 else "",
                                                                      round(massif_to_strength[m] * massif_to_constant[m], 1),
                                                                      AbstractGevTrendTest.nb_years_for_quantile_evolution)
                                    for m in massif_to_strength}
        else:
            massif_name_to_value = massif_to_year
        self.study.visualize_study(None, massif_name_to_color=massif_to_color, show=False,
                                   show_label=False, scaled=True, add_text=add_text,
                                   massif_name_to_value=massif_name_to_value,
                                   fontsize=4)

        title = self.set_trend_test_reparition_title(subtitle, set=True)


        return title

    def set_trend_test_reparition_title(self, subtitle, set=True):
        # Global information
        title = 'Repartition of {} trends'.format(subtitle)
        title += ' at altitude={}m \nfor the starting_year={}'.format(self.altitudes[0], self.first_starting_year)
        if len(self.starting_years) > 1:
            title += ' until starting_year={}'.format(self.last_starting_year)
        title += ' with {} test'.format(get_display_name_from_object_type(self.trend_test_class))
        if self.reduce_strength_array:
            title += '\nEvolution of the quantile {} every {} years'.format(AbstractGevTrendTest.quantile_for_strength,
                                                                            AbstractGevTrendTest.nb_years_for_quantile_evolution)
        if set:
            plt.suptitle(title)
        return title

    def load_axes_for_trend_test_repartition(self, nb_rows, nb_columns=None):
        if nb_columns is None:
            nb_columns = len(self.display_trend_type_to_style)
        fig, axes = plt.subplots(nb_rows, nb_columns, figsize=self.study_visualizer.figsize)
        if isinstance(axes, np.ndarray):
            axes = axes.reshape((nb_rows, nb_columns))
        return axes

    @property
    def altitude_index_level(self):
        return 0

    @property
    def massif_index_level(self):
        return 1

    @staticmethod
    def year_reduction(df, **kwargs):
        # Take the mean with respect to all the first axis indices
        return df.mean(axis=0)

    def visualize_year_trend_test(self, axes=None, marker='o', add_detailed_plots=False, plot_title=None,
                                  isin_parameters=None,
                                  show_or_save_to_file=True,
                                  subtitle_specified=None,
                                  poster_plot=False):
        if axes is None:
            axes = self.load_trend_test_evolution_axes(self.nb_rows)
        else:
            assert len(axes) == self.nb_rows

        results = []
        for idx_reduction, (subtitle, reduction_function) in enumerate(
                self.subtitle_to_reduction_function(self.year_reduction,
                                                    add_detailed_plot=add_detailed_plots,
                                                    subtitle=subtitle_specified).items()):
            specific_title, best_year = self.visualize_trend_test_evolution(
                reduction_function=reduction_function,
                xlabel=STARTING_YEARS_XLABEL,
                xlabel_values=self.starting_years, axes=axes,
                marker=marker,
                subtitle=subtitle,
                isin_parameters=isin_parameters,
                plot_title=plot_title,
                idx_reduction=idx_reduction,
                poster_plot=poster_plot
            )
            results.append((specific_title, best_year, subtitle))
        if show_or_save_to_file:
            last_specific_title = results[-1][0]
            self.show_or_save_to_file(specific_title=last_specific_title,
                                      )
        return results

    @staticmethod
    def index_reduction(df, level):
        # Take the sum with respect to the years, replace any missing data with np.nan
        df = df.any(axis=1)
        # Take the mean with respect to the level of interest
        return df.mean(level=level)

    def visualize_altitude_trend_test(self, axes=None, marker='o', add_detailed_plots=False, plot_title=None,
                                      isin_parameters=None,
                                      show_or_save_to_file=True):
        if axes is None:
            axes = self.load_trend_test_evolution_axes(self.nb_rows)
        else:
            assert len(axes) == self.nb_rows

        last_title = ''
        for subtitle, reduction_function in self.subtitle_to_reduction_function(self.index_reduction,
                                                                                level=self.altitude_index_level,
                                                                                add_detailed_plot=add_detailed_plots).items():
            last_title = self.visualize_trend_test_evolution(reduction_function=reduction_function,
                                                             xlabel=ALTITUDES_XLABEL,
                                                             xlabel_values=self.altitudes, axes=axes, marker=marker,
                                                             subtitle=subtitle, isin_parameters=isin_parameters,
                                                             plot_title=plot_title)
        if show_or_save_to_file:
            self.show_or_save_to_file(specific_title=last_title)
        return last_title

    def visualize_massif_trend_test(self, axes=None, add_detailed_plots=False, plot_title=None,
                                    isin_parameters=None,
                                    show_or_save_to_file=True):
        if axes is None:
            axes = self.load_axes_for_trend_test_repartition(self.nb_rows)
        else:
            assert len(axes) == self.nb_rows

        last_title = ''
        for subtitle, reduction_function in self.subtitle_to_reduction_function(self.index_reduction,
                                                                                level=self.massif_index_level,
                                                                                add_detailed_plot=add_detailed_plots).items():
            last_title = self.visualize_trend_test_repartition(reduction_function, axes, subtitle=subtitle,
                                                               isin_parameters=isin_parameters,
                                                               plot_title=plot_title)
        if show_or_save_to_file:
            self.show_or_save_to_file(specific_title=last_title)

        return last_title

    def visualize_massif_trend_test_one_altitude(self, axes=None, add_detailed_plots=False, plot_title=None,
                                                 isin_parameters=None,
                                                 show_or_save_to_file=True):
        last_title = ''
        for subtitle, reduction_function in self.subtitle_to_reduction_function(self.index_reduction,
                                                                                level=self.massif_index_level,
                                                                                add_detailed_plot=add_detailed_plots).items():
            last_title = self.visualize_trend_test_repartition_poster(reduction_function, axes, subtitle=subtitle,
                                                                      isin_parameters=isin_parameters,
                                                                      plot_title=plot_title)
        if show_or_save_to_file:
            self.show_or_save_to_file(specific_title=last_title, dpi=1000)

        return last_title
