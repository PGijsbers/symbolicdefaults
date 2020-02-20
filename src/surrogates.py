import logging
import os
import pickle
from typing import List, Dict

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from persistence import load_results_for_problem

log = logging.getLogger(__name__)


def load_or_train_surrogates(problem: Dict):
    """ Load surrogate from file if available, train from scratch otherwise. """
    surrogate_file, hyperparameters, performance_column = problem['surrogates'], problem['hyperparameters'], problem['performance_column']

    if os.path.exists(surrogate_file):
        logging.info("Loading surrogates from file.")
        with open(surrogate_file, 'rb') as fh:
            return pickle.load(fh)

    logging.info("Loading experiment results from file.")
    experiments = load_results_for_problem(problem)

    logging.info("Creating surrogates.")
    surrogates = _create_surrogates(
        experiments,
        performance_column=performance_column,
        hyperparameters=hyperparameters,
        metalearner=lambda: RandomForestRegressor(n_estimators=100, n_jobs=-1)
    )
    _save_surrogates(surrogates, surrogate_file)
    return surrogates


def _create_surrogates(
    results: pd.DataFrame,
    performance_column: str,
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
                 f"Creating surrogate for task {task}.")

        task_results = results[results.task_id == task]
        x, y = task_results[hyperparameters], task_results.loc[:,performance_column].values

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
        surrogate.set_params(n_jobs=1)

    with open(output_file, 'wb') as fh:
        pickle.dump(surrogates, fh)
