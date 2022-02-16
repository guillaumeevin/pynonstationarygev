## PyNonStationaryGev

**1. Clone this git project on your computer**

**2. Install required python packages in a virtualenv**

This project relies on Python3.6, if you do not have it, install it and rely on this python for the following.

You can use the integrated tool of the pycharm IDE to create this virtualenv from the requirements.txt files

or you can use the following command in a terminal located at the root of the project:

$ virtualenv <env_name>

$ source <env_name>/bin/activate

(<env_name>)$ pip install -r requirements.txt

**3. Download files**

Several metadata need to be downloaded. You can find this metadata in the following google drive folder ( https://drive.google.com/drive/folders/1bZmmYhyvSqlrgAYXnsF_J2hHdgR41ayl?usp=sharing ). Download all the zip files, unzip them, and put them in the "data" folder.


**4. Generate plots from the data section**

run main_data.py to obtain the plot with the 21 time series with many colors
run main_temperature.py to obtain the plot with the global mean temperatures with respect to the years 

These two scripts are located in the folder projected_extremes/section_data

**5. Generate plots from the results section**

Activate the virtualenv $ source <env_name>/bin/activate

_First step: Select the setting_

In each script, you have to specify two arguments "fast" and "snowfall".
- "fast=False" considers all ensemble members and all elevations, while "fast=True" considers only 6 ensemble mmebers and 1 elevation
- "snowfall=True" corresponds to daily snowfall, while "snowfall=False" corresponds to accumulated ground snow load, "snowfall=None" corresponds to daily winter precipitation

_Second step: The validation experiment_

- run main_model_as_truth_experiment.py for the model as truth experiment (to select the optimal number of linear pieces)
- run main_calibration_validation_experiment_optimized.py for the calibration validation experiment (to select the parameterization for the adjustment coefficients)

These two scripts are located in the folder projected_extremes/section_results/validation_experiment

_Third step: Create some plots_

- main_simple_visualization_with_adjustments.py to create the Figure 4 for the ESD article 
- main_projections_map.py to create maps of return levels, and changes of return levels 
- main_projections_elevation_plot.py to create almost all the Figures for the chapter on extreme snowfall in the PhD manuscript 

These three scripts are located in the folder projected_extremes/section_results
