import functools
import json
import operator
import pickle

import arff
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """ Load arff-file to pandas dataframe. """
    with open(file_path, 'r') as fh:
        data = arff.load(fh)
    return pd.DataFrame(data['data'], columns=[att_name for att_name, att_vals in data['attributes']])


def load_problem(args):
    """ Load the problem configuration specified by `args`. """
    with open(args.config_file, 'r') as fh:
        problem_configurations = json.load(fh)
    configuration = [problem for problem in problem_configurations if problem['name'] == args.problem]

    if len(configuration) < 1:
        raise ValueError("Specified problem '{}' does not exist in {}.".format(args.problem, args.config_file))
    elif len(configuration) > 1:
        raise ValueError("Specified problem '{}' does exists more than once in {}."
                         .format(args.problem, args.config_file))

    return configuration[0]


def load_results_for_problem(problem):
    """ Load the 'experiments' file and filter out those rows that use non-default values. """
    experiments = load_data(problem['experiments'])
    if len(problem['defaults_filters']) > 0:
        filters = [experiments[hp] == default for (hp, default) in problem["defaults_filters"].items()]
        combined_filter = functools.reduce(operator.iand, filters)
        experiments = experiments[combined_filter]
    return experiments


def save_surrogates(surrogates, problem):
    """ Save surrogates to a pickle blob. """
    for surrogate in surrogates.values():
        # We want parallelization during training, but not during prediction
        # as our prediction batches are too small to make the multiprocessing overhead worth it (tested).
        surrogate.set_params(n_jobs=1)

    with open(problem['surrogates'], 'wb') as fh:
        pickle.dump(surrogates, fh)
