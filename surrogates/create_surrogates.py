import logging
from typing import List, Dict

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

log = logging.getLogger(__name__)


def create_surrogates(
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
        log.info("[{:3d}/{:3d}] Creating surrogate for task {}."
                 .format(i+1, len(results.task_id.unique()), task))

        task_results = results[results.task_id == task]
        scores = task_results.predictive_accuracy
        if normalize_scores and (max(scores) - min(scores)) > 0:
            y = (scores - min(scores)) / (max(scores) - min(scores))
        else:
            y = scores

        X = task_results[hyperparameters]
        surrogate_model = metalearner().fit(X, y)

        surrogate_models[int(task)] = surrogate_model

    return surrogate_models
