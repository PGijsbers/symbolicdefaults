import logging
import pickle
import threading
from typing import List, Dict

import pandas as pd

# imports for monkey patching RF:
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._forest import _accumulate_prediction
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted
import numpy as np

log = logging.getLogger(__name__)


def train_save_surrogates(data, hyperparameters, file):
    """ Load surrogate from file if available, train from scratch otherwise. """
    logging.info("Creating surrogates.")
    surrogates = _create_surrogates(
        data,
        hyperparameters=hyperparameters,
        metalearner=lambda: RandomForestRegressor(n_estimators=100, n_jobs=-1)
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
        surrogate.set_params(n_jobs=1)

    with open(output_file, 'wb') as fh:
        pickle.dump(surrogates, fh)


def load_surrogates(surrogate_file):
    logging.info("Loading surrogates from file.")
    with open(surrogate_file, 'rb') as fh:
        surrogates = pickle.load(fh)
        # Even though n_jobs = 1, multiprocessing overhead still occurs
        # We monkey patch it out for ~3x speed up.
        RandomForestRegressor.predict = predict
        return surrogates


def predict(self, X):
    """
    Predict regression target for X.

    The predicted regression target of an input sample is computed as the
    mean predicted regression targets of the trees in the forest.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples. Internally, its dtype will be converted to
        ``dtype=np.float32``. If a sparse matrix is provided, it will be
        converted into a sparse ``csr_matrix``.

    Returns
    -------
    y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        The predicted values.
    """
    check_is_fitted(self)
    # Check data
    X = self._validate_X_predict(X)

    # Assign chunk of trees to jobs
    n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

    # avoid storing the output of every estimator by summing them here
    if self.n_outputs_ > 1:
        y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
    else:
        y_hat = np.zeros((X.shape[0]), dtype=np.float64)

    # Parallel loop
    lock = threading.Lock()
    # <<< sklearn
    # Parallel(n_jobs=n_jobs, verbose=self.verbose,
    #          **_joblib_parallel_args(require="sharedmem"))(
    #     delayed(_accumulate_prediction)(e.predict, X, [y_hat], lock)
    #     for e in self.estimators_)
    # >>> monkey patch
    for e in self.estimators_:
        _accumulate_prediction(e.predict, X, [y_hat], lock)
    # ---------------
    y_hat /= len(self.estimators_)

    return y_hat
