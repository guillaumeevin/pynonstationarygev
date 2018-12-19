


if __name__ == '__main__':
    # Parameters
    scenarios = []
    nb_obs_list = []
    nb_fit = 1000

    # Load the object that will handle the simulation
    simu = Simulations(nb_fit, scenarios, nb_obs_list)

    # Fit many estimators to this simulation
    estimator_types = []
    for estimator_type in estimator_types:
        simu.fit(estimator_type)

    # Comparison of the diverse estimator

    # Compare all the estimator on a global graph (one graph per scenario)
    # On each graph the X axis should be the number of obs
    # the Y graph should the error
    simu.visualize_mean_test_error_graph(estimator_types, scenarios, nb_obs_list)
    # the other possible view, is to have one graph per number of observations
    # on the X axis should the name of the different estimator
    # on the y axis their error


    # Plot the same graph for the train/test error
    # For a single scenario, and a single obs (we give a plot detailing all the estimation steps that enabled to get
    # the result)
    simu.visualize_comparison_graph(estimator_types, scenario, nb_obs)

    # Analyse the result of a single estimator

    # Or all the result could be recorded in a matrix, with scenario as line, and nb_observaitons as columns
    # with the mean value (and the std in parenthesis)
    # (on the border on this matrix we should have the mean value)
    # for example, the first columns should be the mean of the other column for the same line
    simu.visualize_mean_test_error_matrix(estimator_type, scenarios, nb_obs_list)


    #
    simu.visualize



