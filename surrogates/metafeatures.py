import random
import statistics

from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
import openml

from utils import simple_construct_pipeline_for_task


def create_metadataset(task_ids, after_preprocessing=True):
    metafeatures = {}
    for task_id in task_ids:
        task = openml.tasks.get_task(task_id)
        if after_preprocessing:
            pipeline = simple_construct_pipeline_for_task(task)
            metafeatures[task_id] = calculate_metafeatures(task, preprocessing=pipeline)
        else:
            metafeatures[task_id] = calculate_metafeatures(task, preprocessing=None)

    return pd.DataFrame.from_dict(metafeatures, orient='index')


def calculate_metafeatures(task, preprocessing=None):
    X, y = task.get_X_and_y()
    if preprocessing is not None:
        X = preprocessing.fit_transform(X)

    classes, counts = np.unique(y, return_counts=True)
    if preprocessing is not None:
        # For now assume preprocessing always include one-hot encoding.
        n_categorical = len([x for x in X.T if len(set(x)) == 2])
    else:
        n_categorical = len(task.get_dataset().get_features_by_type('nominal', [task.target_name]))

    return dict(
        m=len(set(y)),
        n=X.shape[0],
        p=X.shape[1],
        rc=n_categorical / X.shape[1],
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
    return 1 / statistics.median(distances)
