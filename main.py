import argparse
import functools
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from deap import tools, algorithms

import persistence
from surrogates import create_surrogates
from evolution import setup_toolbox
from evolution.operations import mass_evaluate, n_primitives_in


def main():
    description = "Uses evolutionary optimization to find symbolic expressions for default hyperparameter values."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('problem', type=str,
                        help="Problem to optimize. Must match one of 'name' fields in the configuration file.")
    parser.add_argument('-m',
                        help=("mu for the mu+lambda algorithm. "
                              "Specifies the number of individuals that can create offspring."),
                        dest='mu', type=int, default=20)
    parser.add_argument('-l',
                        help=("lambda for the mu+lambda algorithm. "
                              "Specifies the number of offspring created at each iteration."
                              "Also used to determine the size of starting population."),
                        dest='lambda_', type=int, default=100)
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
    problem = persistence.load_problem(args)

    if os.path.exists(problem['surrogates']):
        logging.info("Loading surrogates from file.")
        with open(problem['surrogates'], 'rb') as fh:
            surrogates = pickle.load(fh)
    else:
        logging.info("Loading experiment results from file.")
        experiments = persistence.load_results_for_problem(problem)

        logging.info("Creating surrogates.")
        surrogates = create_surrogates(
            experiments,
            hyperparameters=problem['hyperparameters'],
            metalearner=lambda: RandomForestRegressor(n_estimators=100, n_jobs=-1)
        )

        persistence.save_surrogates(surrogates, problem)

    # The 'toolbox' defines all operations, and the primitive set (`pset`) defines the grammar.
    toolbox, pset = setup_toolbox(problem)

    # ================================================
    # Start evolutionary optimization
    # ================================================
    metadataset = pd.read_csv(problem['experiment_meta'], index_col=0)
    top_5s = {}

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
            population=toolbox.population(n=args.lambda_),
            toolbox=toolbox,
            mu=args.mu,  # Number of Individuals to pass between generations
            lambda_=args.lambda_,  # Number of offspring per generation
            cxpb=0.5,
            mutpb=0.5,
            ngen=args.ngen,
            verbose=True,
            stats=mstats,
            halloffame=hof
        )
        top_5s[task] = hof[:5]
        logging.info("Top 5 for task {}:".format(task))
        for ind in hof[:5]:
            print(str(ind))


if __name__ == '__main__':
    main()
