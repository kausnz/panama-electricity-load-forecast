#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[511]:


import pandas
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns


# In[512]:


data_file = 'data/continuous_dataset.csv'
df = pd.read_csv(data_file)

data_file = 'data/weekly_pre_dispatch_forecast.csv'
df_forecast_pre_dispatch = pd.read_csv(data_file)


# In[513]:


# This dataset contains the feature variables and dependent variable datetime.
df


# In[514]:


# Load forecast in the pre dispatch reports is the prediction made by the grid operator.
# This is not a feature for our model but could be compared with our predictions as an exploratory activity.
df_forecast_pre_dispatch


# In[515]:


df.describe()


# In[516]:


# Checking if there are missing values. None found.
df.isna().sum()


# In[517]:


# Making columns lower case for better readability.
df.columns = df.columns.str.lower()
df


# In[518]:


# Check the data types of the variables
df.dtypes


# In[519]:


# Checking if variable `school` has non-boolean values
df.school.value_counts()


# In[520]:


# Checking if variable `holiday` has non-boolean values
df.holiday.value_counts()


# Checking if 0's in `holiday_id` matches the number of holidays based on `holiday`.

# In[521]:


assert df.holiday_id.value_counts()[0] == df.holiday.value_counts()[0]


# In[522]:


# datetime is a string. Splitting it to multiple columns will make plotting easier.
# Therefore, creating a new variable called `dt` of type pd.datetime by converting the values from `df.datetime`.
df['dt'] = pd.to_datetime(df.datetime, format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Checking if there are datetime conversion errors.
assert df.dt.isnull().sum() == 0

# delete datetime from the dataframe as dt supersedes it now.
del df['datetime']


# In[523]:


df


# In[524]:


df['dt_year'] = df['dt'].dt.year
df['dt_month'] = df['dt'].dt.month
df['dt_day'] = df['dt'].dt.day
df['dt_hour'] = df['dt'].dt.hour

# No need to separate minute and second values as they are always 0. Verified and confirmed.
# df['dt_minute'] = df.datetime.dt.minute
# df['dt_second'] = df.datetime.dt.second


# In[525]:


# Visually check all dt variables that they've been split correctly from a semantic viewpoint.
# Programmatic check was done above by checking for conversion errors.
df[df.columns[df.columns.str.match('^dt.*')]][::100]


# In[526]:


df.groupby(by=['dt_year', 'dt_month']).size()


# In[527]:


# df[['t2m_toc', 't2m_san', 't2m_dav', 'dt_year', 'dt_month']].groupby(by=['dt_year', 'dt_month'], group_keys=True).max().plot().bar()


# In[528]:


# distribution of the target variable
# %matplotlib inline
# plt.figure(figsize=(10,10))
# sns.histplot(df.nat_demand)
# sns.histplot(y_pred, color='red', bins=50, alpha=0.5)

df.dtypes


# In[529]:


df.nat_demand


# In[530]:


sns.histplot(df.nat_demand, color='red', bins=50, alpha=0.5)


# In[531]:


# See the distribution of the log1p version of the target variable
sns.histplot(np.log1p(df.nat_demand), color='red', bins=50, alpha=0.5)


# 

# In[532]:


# Focusing on the long tail on the left. They seem to be outliers.
plt.xlim(0, 1000)
plt.ylim(0, 10)
sns.histplot(df.nat_demand, color='red', bins=50, alpha=0.5)


# In[533]:


# Looks like there are 8 very low demands. This is only a 0.017% of the total records.
# Without these outliers the national demand values are 'normally' distributed.
# Therefore, no need to log1p() the values.
round(df.nat_demand[df.nat_demand < 600].size / len(df.nat_demand) * 100, 3)


# In[534]:


sns.histplot(df.nat_demand[df.nat_demand > 600], color='red', bins=50, alpha=0.5)


# In[535]:


# Splitting the dataset to 80%, 20%, 20% for training, validation, and testing, respectively.
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
print(f'train:val:test split is {(len(df_train), len(df_val), len(df_test))}')

# Also create a copy of the complete dataframe with order intact to plot actual vs forecast based on the final model.
df_full = df.copy()
print(f'complete dataset is {len(df_full)}')


# In[536]:


# Resetting indices
df_full_train.reset_index(inplace=True)
df_train.reset_index(inplace=True)
df_val.reset_index(inplace=True)
df_test.reset_index(inplace=True)
df_full.reset_index(
    inplace=True)  # not necessary, but done so the for-loop below doesn't fail due to not having an index column.


# In[537]:


y_full_train = df_full_train.nat_demand
y_train = df_train.nat_demand
y_val = df_val.nat_demand
y_test = df_test.nat_demand
y_full = df_full.nat_demand

# log1p
# y_train = np.log1p(df_train.nat_demand)
# y_val = np.log1p(df_val.nat_demand)
# y_test = np.log1p(df_test.nat_demand)


# In[538]:


df_full


# In[539]:


# Removing unwanted variables
for c in ['nat_demand', 'dt', 'index']:
    del df_full_train[c]
    del df_train[c]
    del df_val[c]
    del df_test[c]
    del df_full[c]


# In[541]:


# Vectorize the features
dv = DictVectorizer(sparse=False)
train_dicts = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)

test_dicts = df_test.to_dict(orient='records')
X_test = dv.transform(test_dicts)


# # Random Forest
# 
# Training a Random Forest Regressor by tuning 3 of its parameters, viz. n_estimators, max_depth and min_samples_leaf, to derive the best (lowest) RMSE score. The resulting RMSE will be set as the baseline. Once the baseline is set, an XGBoost Regressor will be trained and evaluated to see if we can achieve a model with a better RMSE.

# ### Benchmark 1
# Train a model and measure performance with defaults n_estimators, max_depth and min_samples_leaf values.

# In[459]:


rf = RandomForestRegressor(n_estimators=100,
                           max_depth=None,
                           min_samples_leaf=1,
                           random_state=1,
                           n_jobs=-1)
model = rf.fit(X_train, y_train)
y_val_pred = rf.predict(X_val)


# In[460]:


rf_performance = [('rmse', np.sqrt(mean_squared_error(y_val, y_val_pred))),
                  ('mae', mean_absolute_error(y_val, y_val_pred))]
pd.DataFrame(rf_performance, columns=['metric', 'score'])


# ### Tune `n_estimators` and `max_depth`

# In[461]:


# Finding the optimal max_depth and n_estimators
scores = []
for d in tqdm([20, 25, 30, 35, 40, 45, 50]):
    for n in tqdm(range(10, 201, 20)):
        rf = RandomForestRegressor(n_estimators=n,
                                   max_depth=d,
                                   random_state=1,
                                   n_jobs=-1)
        rf.fit(X_train, y_train)
        y_val_pred = rf.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        mae = mean_absolute_error(y_val, y_val_pred)
        scores.append((n, d, rmse, mae))


# In[462]:


df_scores = pd.DataFrame(scores, columns=['n_estimators', 'max_depth', 'rmse', 'mae'])
df_scores


# In[463]:


# plt.figure(figsize=(8, 6))
plt.xlabel('No. of estimators')
plt.ylabel('RMSE')
for d in tqdm([20, 25, 30, 35, 40, 45, 50]):
    plt.plot(df_scores[df_scores.max_depth == d].n_estimators,
             df_scores[df_scores.max_depth == d].rmse,
             label=f'max_depth={d}')
plt.legend()


# ### Tune `min_samples_leaf`

# In[464]:


# Based on the above graph, optimal max_depth is 30.
max_depth = 30

# Finding the optimal min_samples_leaf
scores = []
for s in tqdm([1, 3, 5, 10, 50]):
    for n in tqdm(range(10, 201, 20)):
        rf = RandomForestRegressor(n_estimators=n,
                                   max_depth=max_depth,
                                   min_samples_leaf=s,
                                   random_state=1,
                                   n_jobs=-1)
        rf.fit(X_train, y_train)
        y_val_pred = rf.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        mae = mean_absolute_error(y_val, y_val_pred)
        scores.append((n, s, rmse, mae))


# In[465]:


df_scores = pd.DataFrame(scores, columns=['n_estimators', 'min_samples_leaf', 'rmse', 'mae'])
df_scores


# In[466]:


# plt.figure(figsize=(10, 8))
plt.xlabel('No. of estimators')
plt.ylabel('RMSE')
for s in [1, 3, 5, 10, 50]:
    plt.plot(df_scores[df_scores.min_samples_leaf == s].n_estimators,
             df_scores[df_scores.min_samples_leaf == s].rmse,
             label=f'min_samples_leaf={s}')
plt.legend()


# ### Benchmark 2
# Training the model using the optimal parameter values from the above assessment.

# In[467]:


n_estimators = 150
max_depth = 30
min_samples_leaf = 1

rf = RandomForestRegressor(n_estimators=n_estimators,
                           max_depth=max_depth,
                           min_samples_leaf=min_samples_leaf,
                           random_state=1,
                           n_jobs=-1)
rf.fit(X_train, y_train)
y_val_pred = rf.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae = mean_absolute_error(y_val, y_val_pred)

rf_performance = [('rmse', rmse),
                  ('mae', mae)]
pd.DataFrame(rf_performance, columns=['metric', 'score'])


# **Result:**
# Benchmark 2 results is slightly better than Benchmark 1. Therefore, Benchmark 2 will be used as the benchmark for the model performance and will be used for comparison when measuring performance of the gradient boosting models in the next section.

# # XGBoost

# ### Tune `eta`

# In[32]:


features = dv.get_feature_names_out()
dm_train = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dm_val = xgb.DMatrix(X_val, label=y_val, feature_names=features)
dm_test = xgb.DMatrix(X_test, label=y_test, feature_names=features)


# In[560]:


def train_gb_model(dm_train,
                   eta=0.3,
                   max_depth=6,
                   min_child_weight=1,
                   num_boost_round=201,
                   watchlist=[(dm_train, 'train'), (dm_val, 'val')]):
    xgb_params = {
        'eta': eta,
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,

        'eval_metric': 'rmse',
        'objective': 'reg:squarederror',
        'nthread': -1,

        'seed': 1,
        'verbosity': 1
    }
    evals_result = {}
    model = xgb.train(params=xgb_params,
                      dtrain=dm_train,
                      num_boost_round=num_boost_round,
                      evals=watchlist,
                      evals_result=evals_result,
                      verbose_eval=False)

    columns = ['eta', 'iter', 'train_rmse', 'val_rmse']
    train_rmse_scores = list(evals_result['train'].values())[0]
    val_rmse_scores = list(evals_result['val'].values())[0]

    df_scores = pd.DataFrame(
        list(zip([eta] * len(train_rmse_scores),
                 range(1, len(train_rmse_scores) + 1),
                 train_rmse_scores,
                 val_rmse_scores
                 )), columns=columns)
    return model, df_scores


# In[180]:


scores = pd.DataFrame()
for eta in tqdm([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 1.0]):
    key = f'eta={eta}'
    _, df_scores = train_gb_model(dm_train,
                                  eta=eta,
                                  num_boost_round=201)
    scores = pd.concat([scores, df_scores])


# In[184]:


scores.sort_values(by='val_rmse', ascending=True).reset_index().iloc[::200]


# In[218]:


fig, axs = plt.subplots(1, 2)

fig.set_figwidth(20)

axs[0].set_title('Learning Rate - RMSE')
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('RMSE (validation dataset)')
gs = scores.groupby('eta')
gs.get_group(1.00)
gs.groups.values()
for eta in gs.groups.keys():
    df = gs.get_group(eta)
    axs[0].plot(df.iter, df.val_rmse, label=f'eta={eta}')
    axs[0].legend()

axs[1].set_title('Learning Rate - RMSE (Zoomed)')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('RMSE (validation dataset)')
axs[1].set_xlim([175, 200])
axs[1].set_ylim([75, 80])
gs = scores.groupby('eta')
gs.get_group(1.00)
gs.groups.values()
for eta in gs.groups.keys():
    df = gs.get_group(eta)
    axs[1].plot(df.iter, df.val_rmse, label=f'eta={eta}')
    axs[1].legend()


# In[221]:


# Base on the above analysis, eta=0.3 gives the best performance as the learning rate.
chosen_eta = 0.3


# ### Tune `max_depth`

# In[227]:


scores = {}
for max_depth in [3, 4, 6, 10, 14, 18]:
    key = f'max_depth={max_depth}'
    _, scores[key] = train_gb_model(dm_train,
                                    eta=chosen_eta,
                                    max_depth=max_depth,
                                    num_boost_round=201)


# In[229]:


fig, axs = plt.subplots(1, 2)

fig.set_figwidth(20)

axs[0].set_title('Max Depth - RMSE')
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('RMSE (validation dataset)')
for key, df_scores in scores.items():
    axs[0].plot(df_scores.iter, df_scores.val_rmse, label=key)
    axs[0].legend()

axs[1].set_title('Max Depth - RMSE (Zoomed)')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('RMSE (validation dataset)')
axs[1].set_xlim([175, 200])
axs[1].set_ylim([70, 100])
for key, df_scores in scores.items():
    axs[1].plot(df_scores.iter, df_scores.val_rmse, label=key)
    axs[1].legend()


# In[231]:


# The above analysis shows max_depth=10 gies the best performance.
chosen_max_depth = 10


# ### Tune `min_child_weight`

# In[241]:


scores = {}
for min_child_weight in [1, 10, 30, 40]:
    key = f'min_child_weight={min_child_weight}'
    _, scores[key] = train_gb_model(dm_train,
                                    eta=chosen_eta,
                                    max_depth=chosen_max_depth,
                                    min_child_weight=min_child_weight,
                                    num_boost_round=201)


# In[542]:


fig, axs = plt.subplots(1, 2)

fig.set_figwidth(20)

axs[0].set_title('Min Child Weight - RMSE')
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('RMSE (validation dataset)')
for min_child_weight, df in scores.items():
    df = scores[min_child_weight]
    axs[0].plot(df.iter, df.val_rmse, label=min_child_weight)
    axs[0].legend()

axs[1].set_title('Min Child Weight - RMSE (Zoomed)')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('RMSE (validation dataset)')
axs[1].set_xlim([175, 200])
axs[1].set_ylim([70, 73])
for min_child_weight, df in scores.items():
    df = scores[min_child_weight]
    axs[1].plot(df.iter, df.val_rmse, label=min_child_weight)
    axs[1].legend()


# In[243]:


# The above analysis shows min_child_weight=30 gives the best performance.
chosen_min_child_weight = 30


# ### Final GB model

# In[556]:


# Training the model with train set and the chosen values for the parameters

num_boost_round = 200

(model, scores) = train_gb_model(dm_train = dm_train,
                                 eta=chosen_eta,
                                 max_depth=chosen_max_depth,
                                 min_child_weight=chosen_min_child_weight,
                                 num_boost_round=201)
gb_rmse = scores.sort_values(by='val_rmse').iloc[0, 3]
print(f'rmse of xgb model on test set = {gb_rmse}')

full_dicts = df_full.to_dict(orient='records')
X_full = dv.transform(full_dicts)
dm_full = xgb.DMatrix(X_full, label=y_full, feature_names=features)

y_full_pred = model.predict(dm_full)

plt.figure(figsize=(20, 8))
load_period = 24 * 14
actual = y_full[:load_period]
predict = y_full_pred[:load_period]
plt.plot(actual.index, list(actual), label='Actual Load')
plt.plot(actual.index, list(predict), color='red', label='Forecast')
plt.xlabel('Hours')
plt.ylabel('Load (MWh)')
plt.legend()


# In[ ]:


# training with full_train and chosen params, and measure the performance
full_train_dicts = df_full_train.to_dict(orient='records')
X_full_train = dv.transform(full_train_dicts)
dm_full_train = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
(model, scores) = train_gb_model(dm_train=dm_full_train,
                                 eta=chosen_eta,
                                 max_depth=chosen_max_depth,
                                 min_child_weight=chosen_min_child_weight,
                                 num_boost_round=201,
                                 watchlist=[(dm_full_train, 'train'), (dm_test, 'val')])
gb_rmse = scores.sort_values(by='val_rmse').iloc[0, 3]
print(f'rmse of xgb model on full_train set = {gb_rmse}')

# We get much better RMSE compared to what we got from train set. Let's plot the full timeseries with predictions from the new model.
y_full_pred = model.predict(dm_full)

plt.figure(figsize=(20, 8))
load_period = 24 * 14
actual = y_full[:load_period]
predict = y_full_pred[:load_period]
plt.plot(actual.index, list(actual), label='Actual Load')
plt.plot(actual.index, list(predict), color='red', label='Forecast')
plt.xlabel('Hours')
plt.ylabel('Load (MWh)')
plt.legend()


# In[ ]:




