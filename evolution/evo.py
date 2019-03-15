import operator
import random
from typing import List, Dict, Tuple
from functools import partial, reduce
import os
import pickle
import warnings
import sys

import arff
import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.ensemble import RandomForestRegressor

import multiprocessing

from deap import creator, gp, tools, base, algorithms


def load_data(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as fh:
        data = arff.load(fh)
    return pd.DataFrame(data['data'], columns=[att_name for att_name, att_vals in data['attributes']])


def create_surrogates(
    results: pd.DataFrame,
    hyperparameters: List[str],
    normalize_scores: bool = True,
    metalearner: callable = RandomForestRegressor,
    return_train_error: bool = False,
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
    training_errors = dict()
    for i, task in enumerate(results.task_id.unique()):
        logging.info("[{:3d}/{:3d}] Creating surrogate for task {}."
                     .format(i+1, len(results.task_id.unique()), task))

        task_results = results[results.task_id == task]
        scores = task_results.predictive_accuracy
        if normalize_scores and (max(scores) - min(scores)) > 0:
            y = (scores - min(scores)) / (max(scores) - min(scores))
        else:
            y = scores

        X = task_results[hyperparameters]
        surrogate_model = metalearner().fit(X, y)

        if return_train_error:
            from sklearn.metrics import mean_squared_error
            from math import sqrt
            from scipy.stats import spearmanr
            y_hat = surrogate_model.predict(X)
            #rmse = sqrt(mean_squared_error(y, y_hat))
            training_errors[int(task)] = spearmanr(y, y_hat)
        surrogate_models[int(task)] = surrogate_model

    if return_train_error:
        return surrogate_models, training_errors
    return surrogate_models


def n_primitives_in(individual):
    return len([e for e in individual if isinstance(e, gp.Primitive)])


def mass_evaluate(evaluate_fn, individuals, pset, metadataset: pd.DataFrame, surrogates: Dict[str, object]):
    """ evaluate_fn is only for compatability with `map` signature. """
    fns = [gp.compile(individual, pset) for individual in individuals]
    lengths = [max(n_primitives_in(individual), 0) for individual in individuals]
    scores_full = np.zeros(shape=(len(individuals), len(metadataset)), dtype=float)

    for i, ((idx, row), surrogate) in enumerate(zip(metadataset.iterrows(), surrogates)):
        def try_else_invalid(fn, input_):
            try:
                values = fn(**dict(row))
                if any([(isinstance(val, complex) or not np.isfinite(np.float32(val))) for val in values]):
                   raise ValueError("One or more values invalid for input as hyperparameter.")
                return values
            except:
                return (-1e6, -1e6)

        hyperparam_values = [try_else_invalid(fn, row) for fn in fns]

        scores = surrogate.predict(hyperparam_values)

        scores_full[:, i] = scores

    scores_mean = scores_full.mean(axis=1)  # row wise
    return zip(scores_mean, lengths)


def evaluate(individual, pset, metadataset: pd.DataFrame, surrogates: Dict[str, object]):
    """ Evaluate an individual by its projected score across datasets.
    :param individual:
        A compileable DEAP individual.
    :param metadataset:
        A dataframe with task as index and input variables for columns.
    :param surrogates:
        A dictionary that maps a task id to its surrogate model.
    :return:
        The average projected score of the individual.
    """
    fn = gp.compile(individual, pset)
    loose_length = max(n_primitives_in(individual), 0)  # -4
    #bounds = [(0.01, 2.0), (50, 500), (1, 10)]
    scores = []
    for i, row in metadataset.iterrows():
        # We require the metadataset column names to match our variables.
        try:
            hyperparam_values = fn(**dict(row))
            pred_score = surrogates[row.name].predict([[*hyperparam_values]])[0]
            pred_score = random.random()
            #if (not any([isinstance(val, complex) for val in hyperparam_values])
            #        and not all([np.isfinite(np.float32(val)) for val in hyperparam_values])):
            #    raise ValueError("One or more values invalid for input as hyperparameter.")
        #except (ValueError, FloatingPointError, OverflowError) as e:
        except Exception as e:
            # Error because of floating point underflow/overflow/div0 or not valid as input for model.
            logging.warning(str(e))
            return -1, loose_length


        if False:
            penalties = []
            for ((lb, ub), val) in zip(bounds, hyperparam_values):
                d = ub - lb
                if val < lb:
                    dv = lb - val
                    penalties.append(1. - min(1., dv / d))
                elif val > ub:
                    dv = val - ub
                    penalties.append(1. - min(1., dv / d))

            penalty_multiplier = reduce(operator.mul, penalties, 1)
        else:
            penalty_multiplier = 1.

        scores.append(penalty_multiplier * pred_score)
    return np.mean(scores), loose_length


def random_0_1():
    return random.random()


def random_1_10():
    return float(random.randint(1, 10))


def random_log():
    return np.random.choice([2**i for i in range(-8, 9)])


if __name__ == '__main__':
    import logging

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    # Configure numpy to report raise all warnings (very usual
    np.seterr(all='raise')

    optimization_problems = dict(
        adaboost=dict(
            filename='../data/adaboost.arff',
            hyperparameters=[
                'adaboostclassifier__learning_rate',
                'adaboostclassifier__n_estimators',
                'adaboostclassifier__base_estimator__max_depth'],
            surrogates='../data/adaboost_surrogates.pkl'

        ),
        SVC=dict(
            filename='../data/svc.arff',
            hyperparameters=[
                'svc__C',
                'svc__gamma'
            ],
            surrogates='../data/svc_surrogates_1.pkl'
        )
    )
    problem = optimization_problems['SVC']
    # Load ARFF data
    logging.info("Loading data.")
    results = load_data(problem['filename'])

    # Create surrogate model for each dataset.
    # Keep categorical hyperparameters on default (ignoring imputation strategy):
    #results = results[(results.adaboostclassifier__algorithm == 'SAMME.R')]
    results = results[(results.svc__kernel == 'rbf')]

    if os.path.exists(problem['surrogates']):
        with open(problem['surrogates'], 'rb') as fh:
            surrogates = pickle.load(fh)
    else:
        logging.info("Creating surrogate models.")
        surrogates = create_surrogates(
            results,
            hyperparameters=problem['hyperparameters'],
            metalearner=lambda: RandomForestRegressor(n_estimators=100, n_jobs=1)
        )
        with open(problem['surrogates'], 'wb') as fh:
            pickle.dump(surrogates, fh)

    # Load metadataset
    metadataset = pd.read_csv('../data/pp_metadata.csv', index_col=0)

    # Set variables of our genetic program:
    variables = dict(
        NumberOfClasses='m',
        NumberOfFeatures='p',
        NumberOfInstances='n',
        MedianKernelDistance='mkd',
        MajorityClassPercentage='mcp',
        RatioSymbolicFeatures='rc'  # Number Symbolic / Number Features
    )

    variable_names = list(variables.values())

    pset = gp.PrimitiveSetTyped("SymbolicExpression", [float] * len(variables), Tuple)
    pset.renameArguments(**{"ARG{}".format(i): var for i, var in enumerate(variable_names)})
    pset.addEphemeralConstant("cs", random_0_1, ret_type=float)
    pset.addEphemeralConstant("ci", random_1_10, ret_type=float)
    pset.addEphemeralConstant("clog", random_log, ret_type=float)

    binary_operators = [operator.add, operator.mul, operator.sub, operator.truediv, operator.pow]
    unary_operators = [expit, operator.neg]
    for binary_operator in binary_operators:
        pset.addPrimitive(binary_operator, [float, float], float)
    for unary_operator in unary_operators:
        pset.addPrimitive(unary_operator, [float], float)

    # Finally add a 'root-node' primitive (needed to find three functions at once)
    def make_tuple(*args):
        return tuple([*args])
    pset.addPrimitive(make_tuple, [float] * len(problem['hyperparameters']), Tuple)

    # More DEAP boilerplate...
    creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)

    # Use all cores.
    #pool = multiprocessing.Pool()

    def random_mutation(ind, pset):
        valid_mutations = [
            partial(gp.mutNodeReplacement, pset=pset),
            partial(gp.mutInsert, pset=pset),
            partial(gp.mutEphemeral, mode='all')
        ]

        if n_primitives_in(ind) > 0:
            valid_mutations.append(gp.mutShrink)

        return np.random.choice(valid_mutations)(ind)

    toolbox.register("mutate", random_mutation, pset=pset)

    top_5s = {}
    # Leave one out for 100 and 1000 generations.
    for task in list(metadataset.index)[:1]:
        loo_metadataset = metadataset[metadataset.index != task]
        #toolbox.register("evaluate", evaluate, pset=pset, metadataset=loo_metadataset, surrogates=surrogates)
        def no_evaluate():
            raise NotImplementedError
        toolbox.register("evaluate", function=no_evaluate)
        task_surrogates = [surrogates[idx] for idx in loo_metadataset.index]
        toolbox.register("map", partial(mass_evaluate, pset=pset, metadataset=loo_metadataset, surrogates=task_surrogates))
        #toolbox.register("mass_evaluate", mass_evaluate, pset=pset, metadataset=loo_metadataset, surrogates=surrogates)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_size = tools.Statistics(n_primitives_in)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        hof = tools.HallOfFame(10)

        pop, logbook = algorithms.eaMuPlusLambda(
            population=toolbox.population(n=100),
            toolbox=toolbox,
            mu=20,  # Number of Individuals to pass between generations
            lambda_=100,  # Number of offspring per generation
            cxpb=0.5,
            mutpb=0.5,
            ngen=5,
            verbose=True,
            stats=mstats,
            halloffame=hof
        )
                                  #[, stats, halloffame, verbose])
        top_5s[task] = hof[:5]
        logging.info("Top 5 for task {}:".format(task))
        for ind in hof[:5]:
            print(str(ind))
