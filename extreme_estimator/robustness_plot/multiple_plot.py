from extreme_estimator.robustness_plot.single_plot import SinglePlot


class MultiplePlot(SinglePlot):
    """
    In a Multiple Scalar plot, for each

    Each scalar, will be display on a grid row (to ease visual comparison)
    """

    def __init__(self, grid_column_item, plot_row_item, plot_label_item, nb_samples=1, main_title='',
                 plot_png_filename=None):
        super().__init__(grid_row_item=self.OrdinateItem, grid_column_item=grid_column_item,
                         plot_row_item=plot_row_item, plot_label_item=plot_label_item,
                         nb_samples=nb_samples, main_title=main_title, plot_png_filename=plot_png_filename)
        self.kwargs_single_point_to_errors = {}

    def cached_compute_value_from_kwargs_single_point(self, **kwargs_single_point):
        #  Compute hash
        hash_from_kwargs_single_point = self.hash_from_kwargs_single_point(kwargs_single_point)
        # Either compute the errors or Reload them from cached results
        if hash_from_kwargs_single_point in self.kwargs_single_point_to_errors:
            errors = self.kwargs_single_point_to_errors[hash_from_kwargs_single_point]
        else:
            errors = self.compute_value_from_kwargs_single_point(**kwargs_single_point)
            self.kwargs_single_point_to_errors[hash_from_kwargs_single_point] = errors
        return errors

    def hash_from_kwargs_single_point(self, kwargs_single_point):
        items_except_error = [(k, v) for k, v in kwargs_single_point.items() if k != self.OrdinateItem.name]
        ordered_dict_items_str = str(sorted(items_except_error, key=lambda x: x[0]))
        hash_from_kwargs_single_point = hash(ordered_dict_items_str)
        return hash_from_kwargs_single_point
