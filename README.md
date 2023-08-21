# README

## Overview of the submitted folder and the folder structure

Top level:
- run.sh: The shell script to be used in the workflow.
- eda.ipynb:[^1] The EDA for task. 
- metrics_eval.ipynb:[^1] The prediction evaluations to determine the model that performs best in the evaluation metrics.
- feature_importances.ipynb:[^1] From the model that is chosen from the evaluation, the feature importances of the model are calculated. It provides a view of what features are deemed important to the model.  
- README.md: This file

[^1]: Their corresponding html is included as well. The notebooks read from output_all folder.

Folder: src, data, ouput.

Note:
- Information output files are written using `ConfigParser`. It is because it is easy to be read back in `dict` format.  
- `scikit-learn` is the only ML library used.

### A simple descriptions of the files and scripts. A detailed description can be found below.

--> show the direction of input and output.

`main_get_score_db.py` ---> data/score.db and data/score.pkl<br>
Note: To get db file from url. The db and the corresponding dataframe are saved in data/ directory.

data/score.pkl ---> `main_score.py` ---> prep_df.pkl, score_df_X_test.pkl, score_df_y_test.pkl, score_X_train.pkl, score_X_test.pkl, score_y_train.pkl, score_y_test.pkl, features.conf <br>
Note: Preprocessing and prepare the train-test set.

prep_df.pkl ---> `main_feature_selection.py` ---> score_df_X_test.pkl, score_df_y_test.pkl, score_X_train.pkl, score_X_test.pkl, score_y_train.pkl, score_y_test.pkl<br>
Note: This is optional. It is only needed if feature selection is needed.

score_X_train.pkl, score_y_train.pkl ---> `main_model.py` ---> model_name.pkl, models.conf <br>
Note: model_name is made up of the different models made, e.g. polynomial.pkl. models.conf is **manually** copied to params.conf. This is needed for manual model parameters setting.

model_name.pkl, score_X_test.pkl, score_y_test.pkl ---> `main_prediction.py` ---> prediction_score.conf <br>
Note: To compare prediction score.

score_X_train.pkl, score_y_train.pkl ---> `main_importances.py` ---> importance.conf <br>
Note: This is for inspection of feature importances.

`user_score.py`<br>
Note: independent script that make select features, make models and predictions easily in one step. It makes use of all the above scripts and files.

### Detailed descriptions of files and scripts

**output**: The directory where the train-test sets and various info conf files are saved into.
- After any test, this content of the output directory need to be renamed. Because everytime a test is run, the content will be overwritten.

**src**: all the python scripts
- **my_std_lib.py**: It contains all the frequently used variables and functions throughout the various scripts, e.g. the data and the output directory names.

- **main_get_score_db.py**: The main script to get the sqlite3 db file from the url. The db file is then read into a dataframe. Both the db and the dataframe are stored in data/score.db and data/score.pkl respectively.
    - **my_get_db**: The helper functions to get sqlite3 db from url or local file.

- **main_score.py**: The preprocessing of the raw dataset, then store the preprocessed dataframe into prep_df.pkl. After the preprocessing, the dataset is split using `train_test_set`. The `test_size` is set to 20%. The dataframe of the test set is stored in score_df_X_test.pkl and score_df_y_test.pkl. After that, the train and test dataframe is finally made to numpy ndarray form. The ndarray forms are stored in score_X_train.pkl, score_y_test.pkl, score_y_train.pkl, score_y_train.pkl.
     - **my_preprocessing.py**: It does the actual preprocessing of each features like replace str with lowercase characters, get_dummies for categorical variables and etc. It also save the preprocessed dataframe into prep_df.pkl. The names of features chosen are saved in output/features.conf.
     - **my_prepare_dataset.py**: It converts the dataframe into ndarray and into train-test given.

- **main_feature_selection.py**: This file read from the features.conf. The features can be toggled by True or False. It will then convert the dataframe with the selected features into numpy ndarray.
    - Optional: This step is not necessary if no feature selection is needed.
    - The above main_score.py would have written and prepare all the default features and train-test set.

- **main_model.py**: Using the train-test set from the main_score, it does the actual training of models. The small number of parameters of models are chosen by using `GridSearchCV`. Only a few number of parameters of interest are chosen. Models are save to model_name.pkl. The following output files are produced:
    - **cv_score.conf**: The scoring and `best_params_` from `GridSearchCV` are stored here.
    - **models.conf**: All the parameters of the models trained are saved here.
    - **params.conf**: This is a copy of models.conf. The parameters in this file can be set or changed when used with user_score.py. The parameters changed will be used to train the specified model.
    - **my_model_maker.py**: It workhorse of the main_model.py. It saves all the trained models into model_name.pkl.

- **main_prediction.py**: It loads the model_name.pkl and the pickled X_test and y_test to make predictions.
    - **prediction_score.conf**: The prediction scorings of each models.
    - **my_model_eval.py**: The workhorse of the prediction module.

- **main_importances.py**: It loads the model_name.pkl, the pickled X_train and y_train to compute feature importances. This is useful in analysis to see how different features are used by different models. `inspection.permutation_importance` is used for this purpose.
    - **importance.conf**: The `importances_mean` obtained are stored here.
    - **my_model_eval.py**: The workhorse that does the actual `permutation_importance`. The same module that also does the predictions.

- **user_score.py**: The script that allows options to select models to train. It can also read from features.conf to select which features to include for training. It also read from params.conf to get all the parameters to be passed to instantiate a model.
    - The output are the same as previously described.

## Instructions for executing the pipeline and modifying any parameters and Description of logical steps/flow of the pipeline

The logical flow are as followed:
```
#!/bin/bash
# Get the sqlite3 db from the url
python main_get_score_db.py

# Preprocessed and split into train-test set
python main_score.py

# Optional: This step is not necessary if
# no feature selection is needed.
python main_feature_selection.py

# Make the models
python main_model.py

# Make predictions
python main_prediction.py

# Optional: get the feature importances for analysis
python main_importances.py
```

To select features to be included or excluded and to set parameters of models, the following program is used:
```
python user_score.py
usage: user_score [-h] [-l] [-m [model]] [-f [table_name]] [-p [params]]

Evaluate the prediction scorings. Table_name must be in the path
data/table_name.db. Provide only the table_name, e.g. score. Select features
from output/features.conf. Set features to be selected by True or False. Model
parameters can be changed. The output/params.conf contains the default
parameters which can be changed.

options:
  -h, --help            show this help message and exit
  -l, --list            list the models available
  -m [model], --model [model]
                        provide a model name, default sgd
  -f [table_name], --file [table_name]
                        provide the table name, default score
  -p [params], --params [params]
                        model parameters yes or no, default no

usage: user_score -m polynomial -f score -p no
```

Models available are as followed:
```
python user_score.py -l
List of models
polynomial: polynomial linear regression
svr: vector support machine
forest: random forest
sgd: stochastic gradient descent
knn: k nearest neighbors
output/features.conf: select features by True or False
output/params.conf: set parameters for estimator
```

Note:

**Features selection** can be set in `features.conf`. This file will be read in everytime. There is no need to select an option for setting features. All the features are `True` by default.

**table_name** is an option provided to read in other test db if needed. Currently it only reads in `score.db`. If some other test db is used, please specify the **table_name** only, e.g. 'another_score' for 'another_score.db'.

**params** option is used to check the params.conf. Parameters setting can be changed in this file. If the scipt see `-p yes`, it will read the params.conf and get the parameters for training the specified model.

The instructions for each steps are as followed:
```
#!/bin/bash
# These first two steps will get the db and write the necessary files for later use.
# Necessary: Get the sqlite3 db from the url
python main_get_score_db.py

# Necessary: Preprocess and split into train-test set
# This will produce the features.conf file.
# This will be used for feature selection.
# This will also produce the models.conf file.
# This will be used for setting model parameters.
python main_score.py

# Need to copy the models.conf as params.conf to be used
# as an initial model parameters file.
cp ../output/models.conf ../output/params.conf

# To make a polynomial linear regression models,
# default file=score, default model parameters=no
python user_score.py -m polynomial

# Other examples:
# To make random forest regressor, with model parameters
# setting. Default table is score.db.
python user_score.py -m forest -p yes

# To make stochastic gradient descent model with
# another_table.db file. Provide only table_name.
# Default parameters=no.
python user_score.py -m sgd -f another_table

# To make a k nearest neighbors with model parameter
# setting and another_table.db file.
python user_score.py -m knn -p another_table -p yes

# To make a SVM based regressor, with model parameters
# setting in output/params.conf. Default table is score.db.
python user_score.py -m svr -p yes
```

## Overview of key findings from the EDA and feature engineering

- Missing values are found in `attendance_rate` and `final_test`.
- For missing values in `final_test`, first attempt is to impute the missing values with mean. However, a check on the chart show that this imputed values are introducing articificial values into the data. Therefore, it is to drop the samples with missing values in `final_test`. 
- `sleep_duration`: Is almost a binary class, with mean and median about 8 hours. It influences `final_test` score. Because those with less an 8 hours have generally lower score than those with at least 8 hours of sleep.
- `attendance_rate`: It is almost a binary class with mean of about 95%. It influences `final_test`. Because those with low attendance. Missing values in `attendance_rate` are imputed with median.
- However, during the feature importances analysis, these two features are found to be of low importance.
- Boxplots show that `tuition`, `direct_admission`, `hours_per_week`, `cca` and `learning_style` do influence the mean of the `final_test`.
- `number_of_siblings` has values 0, 1 and 2. It is processed into dummies. Its boxplot shows that it is influencing the mean of the `final_test`
- Features that have no influence on the `final_test` are `gender`, `n_male`, `n_female`, `mode_of_transport`, `bag_color` and `student_id`. These features are excluded from further learning.

### Importances of features (post-training analysis)
After the experiment, with random forest as the final selected model, it is found that, the features that ranks top three are `hours_per_week`, `number_of_siblings_0` and `direct_admission`.

## Described how the features in the dataset are processed (summarized in a table)

|Features | Processing |
| ---     | ---        |
|sleep_time,<br> wake_time,<br> sleep_duration,<br> sleep_enough | Convert into pd.Datetime using pd.to_datetime, get the time_delta to find the duration of sleep. The duration is bin into two categories [0, 1] indicating lack or enough sleep The bin edges are [0, 7, 10] hours of sleep.|
|attendace_rate <br> attendance_enough | Missing values are `fillna` with median value. The attendance is bin into two categories [0, 1] indicating poor or good attendance. The bin edges are [0, Q1, 100.0] percent of attendance_rate. |
|final_test | Rows with missing values of final_test are drop. Imputing it with values cause an artificial relationship with other variables. |
| direct_admission | Replace 'Yes', 'No' with 1 and 0. |
| cca | Change the values into lowercase. The none is one category, the other three values `arts`, `sports` and `clubs` are grouped together into one category. The none and others is coded as 0 and 1 respectively. |
| tuition | The category value 'Yes' and 'Y' are replaced with 'yes', and the 'No' and 'N' are replaced with 'no'. Then, they are coded as 0 and 1 respectively. |
| learning_style | The two categories 'Auditory' and 'Visual' are coded into 0 and 1 respectively.  |
| number_of_siblings | The categories 0, 1 and 2 are one-hot encoded using pd.`get_dummies`.  |
| age  | The age -5, -4, 5 and 6 appears to be mistake. They are changed to 15 and 16 respectively for EDA. This feature is dropped. Not used for training. |
| gender,<br> n_female,<br> n_male,<br> mode_of_transport,<br> bag_color,<br> student_id,<br> index | These features are dropped. Not used for training. |

## Explanation of your choice of models

Using ML library in scikit-learn.

The models selected are:
- linear models: polynomial linear regression, stochastic gradient descent with L2 penalty,
- k nearest neighbors regressor,
- random forest regressor,
- SVM based regressor.

Linear models are chosen for their simplicity. It can serve as a baseline performance to be compared with other models. The coefficients from the linear models can provide a glimpse into the relevance or importance of each of the features. But, linear models are only good when the problem is a linear problem. With GridSearchCV, the polynomial degree=2 is found to be the best degree for this problem. When post-training analysis is perfromed, SGD model, it has the worst performance metrics. It could be due to some inappropriate parameters setting.

K nearest neighbors is easy to understand. The brute force search of the scikit-learn is replaced with kd_tree with a shorter training time. The GridSearchCV found the following:
- the best n_neighbors=7 instead of the default 5.
- the distance p=1 (Manhattan distance) instead of the default 2 (Euclidean distance).

Random forest is an ensemble with many decision trees. It can be quite good at its prediction. This particular dataset with quite a number of [0, 1] and dummies (for number_of_siblings) can be efficient for decision tree to make the decision because it has few values to consider. It is a guess that decision tree can perform quite well and efficient with this kind of samples.

SVM regressor projects sample points to a higher dimension to find the decision boundary. It can be efficient in some problems. With GridSearchCV, `rbf` Gaussian based kernel gave the best cross validation score.

## Evaluation of the models developed, explanation of metrics used in the evaluation

This is a regression problem, so the following metrics are used:
- root mean squared error (RMSE): This is calculated from mean squared error. Mean squared error tends to amplify the errors at the extreme values because it is affected by both the mean and the squared value.  
- mean absolute error: This can be affected by errors at the extreme values,
- median absolute error: This is not affected by errors at the extreme values.

Difference between mean abs error and median abs error can provide information about the error at extreme values. If the two values are small, it indicates that there are fewer errors of extreme values.

RMSE is the most strigent measure. A model that scores the lowest in RMSE is worth further examination.

The three error metrics are used to vote for the model with the lowest error values.  

### Evaluation of models (post-training analysis)
A notebook is used for the score analysis. The score is read from prediction_score.conf.
- The linear models have the highest errors in prediction.
- SVM regressor and knn are somewhat comparable in the error values.
- Random forest wins the majority votes. Its has the lowest error values in mean absolute error and root mean squared error. Thus, it is chosen as the final model for the problem.
