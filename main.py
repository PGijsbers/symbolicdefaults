import argparse
import functools
import json
import logging
import operator
import os
import pickle
import random
import typing

import arff
import numpy as np
import pandas as pd
import scipy.special
from sklearn.ensemble import RandomForestRegressor

from deap import gp, creator, base, tools, algorithms

from surrogates import create_surrogates
from evolution.operations import random_mutation, mass_evaluate, n_primitives_in


def load_data(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as fh:
        data = arff.load(fh)
    return pd.DataFrame(data['data'], columns=[att_name for att_name, att_vals in data['attributes']])


def main():
    description = "Uses evolutionary optimization to find symbolic expressions for default hyperparameter values."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('problem', help="Problem to optimize.", type=str)
    parser.add_argument('-m',
                        help=("mu for the mu+lambda algorithm. "
                              "Specifies the number of individuals that can create offspring."),
                        dest='mu', type=int, default=20)
    parser.add_argument('-l',
                        help=("lambda for the mu+lambda algorithm. "
                              "Specifies the number of offspring created at each iteration."
                              "Also used to determine the size of starting population."),
                        dest='lambda', type=int, default=100)
    parser.add_argument('-ngen',
                        help="Number of generations.",
                        dest='ngen', type=int, default=100)
    parser.add_argument('-c',
                        help="Configuration file.",
                        dest='config_file', type=str, default='problems.json')
    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    # Configure numpy to report raise all warnings (otherwise overflows may go undetected).
    np.seterr(all='raise')

    # ================================================
    # Load or create surrogate models for each problem
    # ================================================
    with open(args.config_file, 'r') as fh:
        problem_configurations = json.load(fh)
    configuration = [problem for problem in problem_configurations if problem['name'] == args.problem]
    if len(configuration) < 1:
        raise ValueError("Specified problem '{}' does not exist in {}.".format(args.problem, args.config_file))
    elif len(configuration) > 1:
        raise ValueError("Specified problem '{}' does exists more than once in {}."
                         .format(args.problem, args.config_file))
    else:
        problem = configuration[0]

    if os.path.exists(problem['surrogates']):
        logging.info("Loading surrogates from file.")
        with open(problem['surrogates'], 'rb') as fh:
            surrogates = pickle.load(fh)
    else:
        logging.info("Loading experiment results from file.")
        experiments = load_data(problem['experiments'])
        if len(problem['defaults_filters']) > 0:
            filters = [experiments[hp] == default for (hp, default) in problem["defaults_filters"].items()]
            combined_filter = functools.reduce(operator.iand, filters)
            experiments = experiments[combined_filter]

        logging.info("Creating surrogates.")
        surrogates = create_surrogates(
            experiments,
            hyperparameters=problem['hyperparameters'],
            metalearner=lambda: RandomForestRegressor(n_estimators=100, n_jobs=-1)
        )

        for surrogate in surrogates.values():
            # We want parallelization during training, but not during prediction
            # as our prediction batches are too small to make the multiprocessing overhead worth it (tested).
            surrogate.set_params(n_jobs=1)

        with open(problem['surrogates'], 'wb') as fh:
            pickle.dump(surrogates, fh)

    # ================================================
    # Set up the evolutionary procedure with DEAP
    # ================================================

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

    pset = gp.PrimitiveSetTyped("SymbolicExpression", [float] * len(variables), typing.Tuple)
    pset.renameArguments(**{"ARG{}".format(i): var for i, var in enumerate(variable_names)})
    pset.addEphemeralConstant("cs", lambda: random.random(), ret_type=float)
    pset.addEphemeralConstant("ci", lambda: float(random.randint(1, 10)), ret_type=float)
    pset.addEphemeralConstant("clog", lambda: np.random.choice([2**i for i in range(-8, 9)]), ret_type=float)

    binary_operators = [operator.add, operator.mul, operator.sub, operator.truediv, operator.pow]
    unary_operators = [scipy.special.expit, operator.neg]
    for binary_operator in binary_operators:
        pset.addPrimitive(binary_operator, [float, float], float)
    for unary_operator in unary_operators:
        pset.addPrimitive(unary_operator, [float], float)

    # Finally add a 'root-node' primitive (needed to find three functions at once)
    def make_tuple(*args):
        return tuple([*args])
    pset.addPrimitive(make_tuple, [float] * len(problem['hyperparameters']), typing.Tuple)

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
    toolbox.register("mutate", random_mutation, pset=pset)

    def no_evaluate():
        raise NotImplementedError
    toolbox.register("evaluate", function=no_evaluate)

    # ================================================
    # Start evolutionary optimization
    # ================================================
    metadataset = pd.read_csv(problem['experiment_meta'], index_col=0)
    top_5s = {}
    # Leave one out for 100 and 1000 generations.
    for task in list(metadataset.index)[:1]:
        loo_metadataset = metadataset[metadataset.index != task]
        toolbox.register("map", functools.partial(mass_evaluate,
                                                  pset=pset, metadataset=loo_metadataset, surrogates=surrogates))

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
        # [, stats, halloffame, verbose])
        top_5s[task] = hof[:5]
        logging.info("Top 5 for task {}:".format(task))
        for ind in hof[:5]:
            print(str(ind))


if __name__ == '__main__':
    main()
