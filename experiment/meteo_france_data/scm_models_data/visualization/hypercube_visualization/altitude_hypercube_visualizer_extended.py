from experiment.meteo_france_data.scm_models_data.abstract_extended_study import AbstractExtendedStudy
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.abstract_hypercube_visualizer import \
    AbstractHypercubeVisualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_hypercube_visualizer import \
    AltitudeHypercubeVisualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_year_hypercube_visualizer import \
    Altitude_Hypercube_Year_Visualizer, AltitudeHypercubeVisualizerBis, AltitudeHypercubeVisualizerWithoutTrendType
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.quantity_altitude_visualizer import \
    QuantityAltitudeHypercubeVisualizer


class AltitudeHypercubeVisualizerExtended(AltitudeHypercubeVisualizer):

    def df_bool(self, display_trend_type, isin_parameters=None):
        df_bool = super().df_bool(display_trend_type)
        # Slice a part of the array
        df_bool = self.isin_slicing(df_bool, isin_parameters)
        return df_bool

    def isin_slicing(self, df, isin_parameters):
        if isin_parameters is not None:
            assert isinstance(isin_parameters, list)
            for isin_parameter in isin_parameters:
                transpose, values, level = isin_parameter
                if transpose:
                    df = df.transpose()
                ind = df.index.isin(values=values, level=level)
                res = df.loc[ind].copy()
                df = res.transpose() if transpose else res
        return df

    def _visualize_meta(self, visualization_function, loading_function, name_to_isin_parameters=None,
                        multiplication_factor_column=None, add_detailed_plot=False):
        assert name_to_isin_parameters is not None, 'this method should not be called directly'

        if multiplication_factor_column is None:
            multiplication_factor_row = len(name_to_isin_parameters)
            all_axes = loading_function(self.nb_rows * multiplication_factor_row)
            multiplication_factor = multiplication_factor_row
        else:
            multiplication_factor_row = len(name_to_isin_parameters) // multiplication_factor_column
            multiplication_factor = multiplication_factor_row * multiplication_factor_column
            all_axes = loading_function(self.nb_rows * multiplication_factor_row, multiplication_factor_column)
        specific_title = ''
        for j, (name, isin_parameters) in enumerate(name_to_isin_parameters.items()):
            axes = all_axes[j::multiplication_factor]
            specific_title = visualization_function(axes, plot_title=name,
                                                    isin_parameters=isin_parameters,
                                                    show_or_save_to_file=False,
                                                    add_detailed_plots=add_detailed_plot)
        self.show_or_save_to_file(specific_title=specific_title)

    # Altitude trends

    def _visualize_altitude_trend_test(self, name_to_isin_parameters=None):
        return self._visualize_meta(visualization_function=self.visualize_altitude_trend_test,
                                    loading_function=self.load_trend_test_evolution_axes,
                                    name_to_isin_parameters=name_to_isin_parameters)

    def visualize_altitute_trend_test_by_regions(self):
        return self._visualize_altitude_trend_test(name_to_isin_parameters=self.region_name_to_isin_parameters)

    @property
    def region_name_to_isin_parameters(self):
        return {region_name: [(False, values, self.massif_index_level)]
                for region_name, values in AbstractExtendedStudy.region_name_to_massif_names.items()}

    # Massif trends

    def _visualize_massif_trend_test(self, name_to_isin_parameters=None):
        return self._visualize_meta(visualization_function=self.visualize_massif_trend_test,
                                    loading_function=self.load_axes_for_trend_test_repartition,
                                    name_to_isin_parameters=name_to_isin_parameters)

    def visualize_massif_trend_test_by_altitudes(self):
        return self._visualize_massif_trend_test(name_to_isin_parameters=self.altitude_band_name_to_isin_parameters)

    @property
    def altitude_band_name_to_values(self):
        return {
            '900m <= alti <= 3000m': self.altitudes,
            '900m <= alti <= 1800m': [900, 1200, 1500, 1800],
            '2100m <= alti <= 3000m': [2100, 2400, 2700, 3000],
        }

        # altitude_band = 1000
        # group_idxs = [a // altitude_band for a in self.altitudes]
        # altitude_band_name_to_values = {'All altitudes': self.altitudes}
        # for group_idx in set(group_idxs):
        #     values = [a for a, i in zip(self.altitudes, group_idxs) if i == group_idx]
        #     altitude_band_name = '{}m <= altitude <={}m'.format(group_idx * altitude_band,
        #                                                         (group_idx + 1) * altitude_band)
        #     altitude_band_name_to_values[altitude_band_name] = values
        # return altitude_band_name_to_values

    @property
    def altitude_band_name_to_isin_parameters(self):
        return {altitude_band_name: [(False, values, self.altitude_index_level)]
                for altitude_band_name, values in self.altitude_band_name_to_values.items()}

    # Year trends

    @property
    def massif_name_and_altitude_band_name_to_isin_parameters(self):
        d = {}
        for massif_name, isin_parameters1 in self.region_name_to_isin_parameters.items():
            for altitude_band_name, isin_parameters2 in self.altitude_band_name_to_isin_parameters.items():
                name = massif_name + ' ' + altitude_band_name
                isin_parameters = isin_parameters1 + isin_parameters2
                d[name] = isin_parameters
        return d

    def vsualize_year_trend_by_regions_and_altitudes(self, add_detailed_plot=False):
        return self._visualize_meta(visualization_function=self.visualize_year_trend_test,
                                    loading_function=self.load_trend_test_evolution_axes_with_columns,
                                    name_to_isin_parameters=self.massif_name_and_altitude_band_name_to_isin_parameters,
                                    multiplication_factor_column=len(self.altitude_band_name_to_isin_parameters),
                                    add_detailed_plot=add_detailed_plot)





class AltitudeHypercubeVisualizerWithoutTrendExtended(AltitudeHypercubeVisualizerExtended,
                                                      AltitudeHypercubeVisualizerWithoutTrendType):

    def df_loglikelihood(self, isin_parameters=None):
        return self.isin_slicing(df=super().df_loglikelihood(), isin_parameters=isin_parameters)



# Extension

class AltitudeHypercubeVisualizerBisExtended(AltitudeHypercubeVisualizerExtended, AltitudeHypercubeVisualizerBis):
    pass

class AltitudeYearHypercubeVisualizerExtended(AltitudeHypercubeVisualizerExtended, Altitude_Hypercube_Year_Visualizer):
    pass


# Quantity hypercube

class QuantityHypercubeWithoutTrend(AltitudeHypercubeVisualizerWithoutTrendType, QuantityAltitudeHypercubeVisualizer):
    pass


class QuantityHypercubeWithoutTrendExtended(AltitudeHypercubeVisualizerWithoutTrendExtended, QuantityHypercubeWithoutTrend):
    pass
