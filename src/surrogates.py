import logging
import pickle
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer

log = logging.getLogger(__name__)

def train_save_surrogates(data, hyperparameters, file):
    """ Load surrogate from file if available, train from scratch otherwise. """
    logging.info("Creating surrogates.")
    
    # Make a preprocessing pipeline
    ftf = make_pipeline(SimpleImputer(strategy='constant'))
    ctf = make_pipeline(SimpleImputer(strategy='constant'))
    prep = make_column_transformer(
        (ftf, data[hyperparameters].select_dtypes([np.number]).columns.values.tolist()),
        (ctf, data[hyperparameters].select_dtypes([object]).columns.values.tolist()))
    clf = make_pipeline(prep, RandomForestRegressor(n_estimators=100, n_jobs=-1))

    surrogates = _create_surrogates(
        data,
        hyperparameters=hyperparameters,
        metalearner=lambda: clf
    )
    _save_surrogates(surrogates, file)
    return surrogates


def _create_surrogates(
    results: pd.DataFrame,
    hyperparameters: List[str],
    normalize_scores: bool = True,
    metalearner: callable = RandomForestRegressor
) -> Dict[int, object]:
    """ For each task (`task_id`) in the dataframe, create a surrogate model.
    The surrogate model will predict (*hyperparameters) -> score.

    :param results: pd.DataFrame
    :param hyperparameters: List[str].
        columnnames for the hyperparameters on which to create predictions.
    :param normalize_scores: bool
        If True, normalize the performance scores per task.
    :param metalearner: callable
        Instantiates a machine learning model that has `fit` and `predict`.
    :return: dict[int, object]
        A dictionary that maps each task id to its surrogate model.
    """
    surrogate_models = dict()
    for i, task in enumerate(results.task_id.unique()):
        log.info(f"[{i+1:3d}/{len(results.task_id.unique()):3d}] "
                 f"Creating surrogate for task {task:6.0f}.")


         
        # Ordinal Encode Categoricals for RandomForest
        cv = results.loc[results.task_id == task, hyperparameters].select_dtypes([object])
        results.loc[results.task_id == task, cv.columns] = OrdinalEncoder().fit_transform(cv.values)
        
        task_results = results[results.task_id == task]
        x, y = task_results[hyperparameters], task_results.target

        if normalize_scores:
            if (max(y) - min(y)) == 0:
                raise RuntimeError(f"Can not normalize scores for task {task}."
                                   f"Min and Max scores are equal.")
            y = (y - min(y)) / (max(y) - min(y))

        surrogate_model = metalearner().fit(x, y)
        surrogate_models[int(task)] = surrogate_model

    return surrogate_models


def _save_surrogates(surrogates, output_file: str):
    """ Save surrogates to a pickle blob. """
    for surrogate in surrogates.values():
        # We want parallel training, but not prediction. Our prediction batches
        # are too small to make the multiprocessing overhead worth it (verified).
        surrogate.set_params(randomforestregressor__n_jobs=1)

    with open(output_file, 'wb') as fh:
        pickle.dump(surrogates, fh)
