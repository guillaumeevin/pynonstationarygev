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

First step: The validation experiment

- run main_model_as_truth_experiment.py for the model as truth experiment (to select the optimal number of linear pieces)
- run main_calibration_validation_experiment_optimized.py for the calibration validation experiment (to select the parameterization for the adjustment coefficients)

Second step: Create some plots
