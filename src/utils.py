import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold


# === OpenML Hackery ===
# One flow can not be a subflow twice, which is inconvenient since
# we use two instantiations of SimpleImputer. Until we change this
# on an OpenML level, we need to kind of work around this if we
# want to use the convenient `run_model_on_task` function.
__version__ = '0.0'


class SimpleImputerDuplicate(SimpleImputer):
    pass
# ==== end hackery ====


def simple_construct_pipeline_for_task(task, learner=None):
    cat_indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])
    num_indices = task.get_dataset().get_features_by_type('numeric', [task.target_name])

    X, y = task.get_X_and_y()

    all_nan = np.where([np.all(np.isnan(v)) for v in X.T])[0]
    cat_indices = [i for i in cat_indices if not i in all_nan]
    num_indices = [i for i in num_indices if not i in all_nan]
    n_cat, n_num = len(cat_indices), len(num_indices)

    unique_values_cat_features = [list(sorted([v for v in set(values) if not np.isnan(v)]))
                                  for values in X[:, cat_indices].T]

    preprocessing_steps = [
                        ("impute", ColumnTransformer(
                            transformers=[
                                ("nominal", SimpleImputer(strategy="most_frequent"), cat_indices),
                                ("numeric", SimpleImputerDuplicate(strategy="mean"), num_indices),
                                ("all_nan", SimpleImputerDuplicate(strategy="constant", fill_value = 0), all_nan)
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
                        ("feature_selection", VarianceThreshold())
                    ]
    learner_step = [('learner', learner)] if learner is not None else []
    return Pipeline(steps=preprocessing_steps + learner_step)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
