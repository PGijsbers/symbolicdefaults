import random
import statistics

import scipy.sparse
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
import openml
import joblib

from src.utils import simple_construct_pipeline_for_task

memory = joblib.Memory("data/cache", verbose = 0)

def create_metadataset(task_ids, after_preprocessing=True):
    metafeatures = {}
    for task_id in task_ids:
        try:
            task = openml.tasks.get_task(task_id)
            if after_preprocessing:
                pipeline = simple_construct_pipeline_for_task(task)
                metafeatures[task_id] = calculate_metafeatures(task, preprocessing=pipeline)
            else:
                metafeatures[task_id] = calculate_metafeatures(task, preprocessing=None)
        except Exception as e:
            print('Skipped task {} because of {}'.format(task_id, str(e)))

    return pd.DataFrame.from_dict(metafeatures, orient='index')


@memory.cache()
def calculate_metafeatures(task, preprocessing=None):
    X, y = task.get_X_and_y()

    # number of features before preprocessing
    po = X.shape[1]

    if preprocessing is not None:
        X = preprocessing.fit_transform(X)

    n = X.shape[0]
    p = X.shape[1]

    if isinstance(X, scipy.sparse.csr_matrix):
        if (n > 1e4): # Random 1e4 rows
            # Sub-sample to avoid running OOM (this is only used for estimating mkd)
            idx = np.random.choice(range(int(n)), size = int(1e4), replace = False)
            X = X[idx,:]
        if (p > 1e4): # First 1e4 cols
            idp = range(int(1e4))
            X = X[:,idp]
        # quick hack for now as code below assumes operations that cant be done on csr matrix (most notably mkd code)
        X = X.todense().A

    classes, counts = np.unique(y, return_counts=True)
    if preprocessing is not None:
        # For now assume preprocessing always include one-hot encoding and does not introduce additional columns
        # for numeric transformations. Might be off if constant features are removed.
        n_categorical = p - len(task.get_dataset().get_features_by_type('numeric', [task.target_name]))
    else:
        n_categorical = len(task.get_dataset().get_features_by_type('nominal', [task.target_name]))

    return dict(
        m=len(set(y)),
        n=n,
        po=po,
        p=p,
        rc=max(0.0, n_categorical / p),
        mcp=max(counts) / len(y),
        mkd=mkd(X),
        xvar=X.var()
    )


def mkd(X):
    # remove rows with any nans, and columns with only nans
    nan_cols = np.isnan(X).all(axis=0)
    X_col_clean = X[:, ~nan_cols]
    nan_rows = np.isnan(X_col_clean).any(axis=1)
    X_no_nan = X_col_clean[~nan_rows, :]


    # scale to N(0, 1)
    scaled = scale(X_no_nan)

    # pick randomly half the remainder, twice
    n = int(X.shape[0] * 0.5)
    row_subset1 = random.choices(scaled, k=n)
    row_subset2 = random.choices(scaled, k=n)

    # calculate their distance
    distances = [sum((x1 - x2)**2) for (x1, x2) in zip(row_subset1, row_subset2)]

    # pick the median
    return 1 / statistics.median([d for d in distances if d != 0])


if __name__ == '__main__':
    create_metadataset([167121.])