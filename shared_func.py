import pandas as pd
import xgboost as xgb


def train_gb_model(dm_train,
                   eta=0.3,
                   max_depth=6,
                   min_child_weight=1,
                   num_boost_round=201,
                   watchlist=None):
    """ Trains an XGBoost regression model

    :param dm_train: training data
    :param eta: model learning rate
    :param max_depth: maximum depth of the decision tree
    :param min_child_weight: Minimum sum of instance weight needed in a child
    :param num_boost_round: the number of iteration for boosting
    :param watchlist: the list of train and validation datasets for evaluation
    :return: a tuple of two elements, model and eval score dataframe, respectively
    """

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
    mdl = xgb.train(params=xgb_params,
                    dtrain=dm_train,
                    num_boost_round=num_boost_round,
                    evals=watchlist,
                    evals_result=evals_result,
                    verbose_eval=False)

    columns = ['eta', 'iter', 'train_rmse', 'val_rmse']
    train_rmse_scores = list(evals_result['train'].values())[0] if watchlist is not None else []
    val_rmse_scores = list(evals_result['val'].values())[0] if watchlist is not None else []

    df_scores = pd.DataFrame(
        list(zip([eta] * len(train_rmse_scores),
                 range(1, len(train_rmse_scores) + 1),
                 train_rmse_scores,
                 val_rmse_scores
                 )), columns=columns)
    return mdl, df_scores
