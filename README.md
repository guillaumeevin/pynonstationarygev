## PyNonStationaryGev

**1. Clone this git project on your computer**

**2. Install required python packages in a virtualenv**

$ virtualenv <env_name>

$ source <env_name>/bin/activate

(<env_name>)$ pip install -r requirements.txt

**3. Download files**

Several metadata need to be downloaded. You can find this metadata in the following google drive folder ( https://drive.google.com/drive/folders/1bZmmYhyvSqlrgAYXnsF_J2hHdgR41ayl?usp=sharing ). Download all the zip files, unzip them, and put them in the "data" folder.

**4. Run the code**

Activate the virtualenv $ source <env_name>/bin/activate

_First step: Select the setting_

In each script, you have to specify two arguments "fast" and "snowfall".
- "fast=False" considers all ensemble members and all elevations, while "fast=True" considers only 6 ensemble mmebers and 1 elevation
- "snowfall=True" corresponds to snowfall, while "snowfall=False" corresponds to ground snow load. 

_Second step: The validation experiment_

- run main_model_as_truth_experiment.py for the model as truth experiment (to select the optimal number of linear pieces)
- run main_calibration_validation_experiment_optimized.py for the calibration validation experiment (to select the parameterization for the adjustment coefficients)

These two scripts are located in the folder projected_extremes/section_results/validation_experiment

_Third step: Create some plots_

-

These scripts are located in the folder projected_extremes/section_results
