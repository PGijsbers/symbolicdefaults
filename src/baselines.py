import argparse
import functools
import logging
import sys
import time
import numpy as np
import pandas as pd

from problem import Problem
from main import configure_logging


def cli_parser():
    description = "Compute baselines for a given problem"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('problem', type=str,
                        help="Problem to optimize. Must match one of 'name' fields in the configuration file.")
    return parser.parse_args()

def get_oracle_performance(problem_data):
    """
    Obtains the oracle performance. This is the "best" performance found in the large random search.
    """
    df = problem_data.groupby("task_id")["target"].max()
    df = pd.DataFrame(df).transpose()
    df.index.name = 'search_type'
    return df.rename(index={'target':'oracle'})


def rep_random_search(problem_data, nrs, replications = 5):
    """
    Simulates repeated random search for 'nrs' iterations, replicating 'replications' times.
    Random search is simulated by sampling from the data.
    Note that this results are optimistic, since the nested-cv-model-selection step is ignored.
    Computes mean and std accross replications for each task
    """
    odf = pd.DataFrame(columns = ['target'])
    for i in range(replications):
        df = problem_data.sample(frac=1).groupby('task_id').head(nrs).loc[:,['task_id', 'target']]
        df = pd.DataFrame(df.groupby('task_id').max())
        odf = odf.append(df)

    odf = odf.groupby(level=0).agg({'target':['mean', 'std']}).transpose()
    odf = odf.rename(index = {'target': f'rs_{nrs}'})
    odf.index.names = ["search_type", "aggregation"]
    return odf

def get_regret(df, odf):
    return np.abs(df - odf.loc['oracle',:])

def normalize_scores(ys):
    y = ys.values
    if (max(y) - min(y)) == 0:
        return pd.Series(np.zeros(shape = len(y)), index = ys.index)
    return pd.Series((y - min(y)) / (max(y) - min(y)), index = ys.index)

def main():
    # Numpy must raise all warnings, otherwise overflows may go undetected.
    np.seterr(all='raise')
    args = cli_parser()
    configure_logging(None)

    for parameter, value in args._get_kwargs():
        logging.info(f"param:{parameter}:{value}")

    problem = Problem(args.problem)
    prob_df = problem.data

    # Normalize scores to [0,1]
    logging.info(f"Scaling to [0,1] (1: best)")
    prob_df['target'] = prob_df.groupby('task_id')['target'].apply(normalize_scores)

    odf = get_oracle_performance(problem.data)
    logging.info(f"Avg. Performance for Oracle: {odf.aggregate('mean', axis=1).values[0]:.5f}")

    for nrs in [2, 4, 8, 16, 32, 64, 128]:
        rsdf = rep_random_search(problem.data, nrs=nrs)
        df = rsdf.query('aggregation == "mean"')
        df.index = df.index.droplevel('aggregation')
        logging.info(f"Avg. Performance for {nrs} iter random search: {df.aggregate('mean', axis=1).values[0]:.5f}")
        # logging.info(f"Avg. Regret for {nrs} iter random search: {get_regret(df, odf).aggregate('mean', axis=1).values[0]:.5f}")
        odf = odf.append(df)

if __name__ == '__main__':
    main()
