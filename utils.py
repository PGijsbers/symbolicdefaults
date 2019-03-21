import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

__version__ = '0.0'


class SimpleImputerDuplicate(SimpleImputer):
    pass


def simple_construct_pipeline_for_task(task, learner):
    cat_indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])
    num_indices = task.get_dataset().get_features_by_type('numeric', [task.target_name])
    n_cat, n_num = len(cat_indices), len(num_indices)

    X, y = task.get_X_and_y()
    unique_values_cat_features = [list(sorted([v for v in set(values) if not np.isnan(v)]))
                                  for values in X[:, cat_indices].T]

    return Pipeline(steps=
                    [
                        ("impute", ColumnTransformer(
                            transformers=[
                                ("nominal", SimpleImputer(strategy="most_frequent"), cat_indices),
                                ("numeric", SimpleImputerDuplicate(strategy="mean"), num_indices)
                            ],
                            remainder="passthrough"
                        )),
                        ("transform", ColumnTransformer(
                            transformers=[
                                ("nominal", OneHotEncoder(categories=unique_values_cat_features), [*range(n_cat)]),
                                ("numeric", StandardScaler(), [*range(n_cat, n_cat + n_num)])
                            ],
                            remainder="passthrough"
                        )),
                        ("feature_selection", VarianceThreshold()),
                        ("clasification", learner)
                    ])
