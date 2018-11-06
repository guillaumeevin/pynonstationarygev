from extreme_estimator.robustness_plot.abstract_robustness_plot import AbstractPlot


class MultipleScalarPlot(AbstractPlot):
    """
    In a Multiple Scalar plot, for each

    Each scalar, will be display on a grid row (to ease visual comparison)
    """

    def __init__(self, grid_column_item, plot_row_item, plot_label_item):
        super().__init__(grid_row_item=None, grid_column_item=grid_column_item,
                         plot_row_item=plot_row_item, plot_label_item=plot_label_item)

