from experiment.meteo_france_data.scm_models_data.abstract_extended_study import AbstractExtendedStudy
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.abstract_hypercube_visualizer import \
    AbstractHypercubeVisualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_hypercube_visualizer import \
    AltitudeHypercubeVisualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_year_hypercube_visualizer import \
    Altitude_Hypercube_Year_Visualizer, AltitudeHypercubeVisualizerBis


class AltitudeHypercubeVisualizerExtended(AltitudeHypercubeVisualizer):

    def df_bool(self, display_trend_type, isin_parameters=None):
        df_bool = super().df_bool(display_trend_type)
        # Slice a part of the array
        if isin_parameters is not None:
            assert isinstance(isin_parameters, list)
            for isin_parameter in isin_parameters:
                transpose, values, level = isin_parameter
                if transpose:
                    df_bool = df_bool.transpose()
                ind = df_bool.index.isin(values=values, level=level)
                res = df_bool.loc[ind].copy()
                df_bool = res.transpose() if transpose else res
        return df_bool

    @property
    def region_name_to_isin_parameters(self):
        return {region_name: [(False, values, self.massif_index_level)]
                for region_name, values in AbstractExtendedStudy.region_name_to_massif_names.items()}

    @property
    def nb_regions(self):
        return len(self.region_name_to_isin_parameters)

    def altitude_band_name_to_isin_parameters(self):
        return self.altitudes

    def visualize_altitute_trend_test_by_regions(self):
        return self._visualize_altitude_trend_test(name_to_isin_parameters=self.region_name_to_isin_parameters)

    def _visualize_altitude_trend_test(self, name_to_isin_parameters=None):
        assert name_to_isin_parameters is not None, 'this method should not be called directly'
        multiplication_factor = len(name_to_isin_parameters)
        all_axes = self.load_trend_test_evolution_axes(self.nb_rows * multiplication_factor)
        specific_title = ''
        for j, (name, isin_parameters) in enumerate(name_to_isin_parameters.items()):
            axes = all_axes[j::multiplication_factor]
            specific_title = self.visualize_altitude_trend_test(axes, plot_title=name, isin_parameters=isin_parameters, show_or_save_to_file=False)
        print(specific_title)
        self.show_or_save_to_file(specific_title=specific_title)


class AltitudeHypercubeVisualizerBisExtended(AltitudeHypercubeVisualizerExtended, AltitudeHypercubeVisualizerBis):
    pass


class AltitudeYearHypercubeVisualizerExtended(AltitudeHypercubeVisualizerExtended, Altitude_Hypercube_Year_Visualizer):
    pass
