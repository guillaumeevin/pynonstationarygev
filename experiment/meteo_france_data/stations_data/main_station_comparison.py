from experiment.meteo_france_data.stations_data.comparison_analysis import ComparisonAnalysis

if __name__ == '__main__':
    comparison = ComparisonAnalysis(altitude=900, nb_border_data_to_remove=nb, margin=150,
                       exclude_some_massifs_from_the_intersection=nb == 2,
                       transformation_class=transformation_class,
                       normalize_observations=True)