import functools
import json
import operator

import arff
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """ Load arff-file to pandas dataframe. """
    with open(file_path, 'r') as fh:
        data = arff.load(fh)
    return pd.DataFrame(data['data'], columns=[att_name for att_name, att_vals in data['attributes']])


def load_problem(problem_name):
    """ Load the problem configuration specified by `args`. """
    with open('problems/{}.json'.format(problem_name), 'r') as fh:
        return json.load(fh)


def load_results_for_problem(problem):
    """ Load the 'experiments' file and filter out those rows that use non-default values. """
    experiments = load_data(problem['rs_data'])
    if len(problem['defaults_filters']) > 0:
        filters = [experiments[hp] == default for (hp, default) in problem["defaults_filters"].items()]
        combined_filter = functools.reduce(operator.iand, filters)
        experiments = experiments[combined_filter]
    if problem.get('ignore') is not None and len(problem['ignore']) > 0:
        experiments = experiments.drop(problem['ignore'], axis=1)
    return experiments
