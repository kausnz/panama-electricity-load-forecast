""" Trains a XGBoost regression model with a predefined set of parameters and exports as a bentoml model package

When training the model it is configured to use the parameters chosen as optimal for the model. The parameter
tuning and selection was done in the notebook.ipynb.

"""
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

import xgboost as xgb

import bentoml

from shared_func import train_gb_model

data_file = 'data/continuous_dataset.csv'

df = pd.read_csv(data_file)

dv = DictVectorizer(sparse=False)

df_full_train, _ = train_test_split(df, test_size=0.2, random_state=1)
y_full_train = df_full_train.nat_demand

full_train_dicts = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dicts)

# retraining without the feature_names in the dmatrix, as otherwise predict() will fail later in the pipeline.
dm_full_train = xgb.DMatrix(X_full_train, label=y_full_train)

# Training the model
(model, scores) = train_gb_model(dm_train=dm_full_train,
                                 eta=0.3,
                                 max_depth=10,
                                 min_child_weight=30,
                                 num_boost_round=201,
                                 watchlist=None)

# Exporting the model as a bentoml package
bentoml.xgboost.save_model("load_forecast_model", model,
                           custom_objects={
                               "dictVectorizer": dv
                           },
                           signatures={
                               "predict": {
                                   "batchable": True,
                                   "batch_dim": 0
                               }
                           })
