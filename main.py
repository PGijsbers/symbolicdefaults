import argparse
import functools
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from deap import tools

import persistence
from surrogates import create_surrogates
from evolution import setup_toolbox
from evolution.operations import mass_evaluate, n_primitives_in, mut_all_constants
from evolution.algorithms import one_plus_lambda, eaMuPlusLambda, random_search

from deap import gp, creator


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
    parser.add_argument('-a',
                        help="Algorithm. {mupluslambda, onepluslambda}",
                        dest='algorithm', type=str, default='mupluslambda')
    parser.add_argument('-s',
                        help="Evaluate individuals on a random [S]ubset of size [0, 1].",
                        dest='subset', type=float, default=1.)
    parser.add_argument('-esn',
                        help="Early Stopping N. Stop optimization if there is no improvement in n generations.",
                        dest='early_stopping_n', type=int, default=20)
    parser.add_argument('-o',
                        help="Output file. Also write log output to this file.",
                        dest='output_file', type=str, default=None)
    parser.add_argument('-mno',
                        help="Max Number of Operators",
                        dest='max_number_operators', type=int, default=None)
    parser.add_argument('-pp',
                        help="Phenotypic Plasticity",
                        dest='phenotypic_plasticity', type=bool, default=False)
    parser.add_argument('-oc',
                        help=("Optimize Constants. Instead of evaluating an individual with specific constants"
                              "evaluate based it on 50 random instantiation of constants instead."),
                        dest='optimize_constants', type=bool, default=False)
    parser.add_argument('-t',
                        help="Perform search and evaluation for this task only.",
                        dest='task', type=int, default=None)
    args = parser.parse_args()

    if args.optimize_constants and args.phenotypic_plasticity:
        raise ValueError("Phenotypic Plasticity together with Optimize Constants currently not supported.")

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    if args.output_file is not None:
        log_file_handle = logging.FileHandler(args.output_file)
        log_file_handle.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(log_file_handle)

    # Configure numpy to report raise all warnings (otherwise overflows may go undetected).
    np.seterr(all='raise')

    # ================================================
    # Load or create surrogate models for each problem
    # ================================================
    problem = persistence.load_problem(args.problem)

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
    toolbox, pset = setup_toolbox(problem, args)

    # ================================================
    # Start evolutionary optimization
    # ================================================
    metadataset = pd.read_csv(problem['metadata'], index_col=0)
    metadataset = metadataset[metadataset.index.isin(surrogates)]
    top_5s = {}

    for task in list(metadataset.index):
        if args.task is not None and args.task != task:
            continue
        logging.info("START_TASK:{}".format(task))
        loo_metadataset = metadataset[metadataset.index != task]
        toolbox.register("map", functools.partial(mass_evaluate,
                                                  pset=pset, metadataset=loo_metadataset,
                                                  surrogates=surrogates, subset=args.subset,
                                                  toolbox=toolbox,
                                                  optimize_constants=args.optimize_constants,
                                                  phenotypic_plasticity=args.phenotypic_plasticity))

        pop = toolbox.population(n=args.lambda_)
        P = pop[0]

        last_best = (0, -10)
        last_best_gen = 0

        # Set up things to track on the optimization process
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_size = tools.Statistics(n_primitives_in)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        hof = tools.HallOfFame(10)

        # Little hackery for logging with early stopping
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + mstats.fields

        for i in range(args.ngen):
            # Hacky way to integrate early stopping with DEAP.
            if args.algorithm == 'mupluslambda':
                pop, _ = eaMuPlusLambda(
                    population=pop,
                    toolbox=toolbox,
                    mu=args.mu,  # Number of Individuals to pass between generations
                    lambda_=args.lambda_,  # Number of offspring per generation
                    cxpb=0.5,
                    mutpb=0.5,
                    ngen=1,
                    verbose=False,
                    halloffame=hof,
                    no_cache=((args.subset < 1.0) or args.phenotypic_plasticity)
                )
            if args.algorithm == 'onepluslambda':
                P, pop = one_plus_lambda(
                    P=P,
                    toolbox=toolbox,
                    ngen=1,
                    halloffame=hof
                )
            if args.algorithm == 'random_search':
                pop = random_search(toolbox, popsize=args.lambda_, halloffame=hof)

            # Little hackery for logging with early stopping
            record = mstats.compile(pop) if mstats is not None else {}
            logbook.record(gen=i, nevals=100, **record)

            generation_info_string = "GEN_{}_FIT_{}_{}_{}_SIZE_{}_{}_{}".format(i,
                                 record['fitness']['min'], record['fitness']['avg'], record['fitness']['max'],
                                 record['size']['min'], record['size']['avg'], record['size']['max'])

            logging.info(generation_info_string)
            # Little hackery for logging early stopping
            for ind in hof:
                if ind.fitness.wvalues > last_best:
                    last_best = ind.fitness.wvalues
                    last_best_gen = i
            if i - last_best_gen > args.early_stopping_n:
                logging.info("Stopping early, no new best in {} generations.".format(args.early_stopping_n))
                break

        top_5s[task] = hof[:5]
        logging.info("Top 5 for task {}:".format(task))
        for ind in hof[:5]:
            if args.phenotypic_plasticity:
                logging.info(str(ind)+str(ind.plasticity))
            elif args.optimize_constants:
                # since 'optimization' of constants is not saved, reoptimize constants before printing.
                variations = [mut_all_constants(toolbox.clone(ind), pset) for _ in range(50)]
                fitnesses = mass_evaluate(toolbox.evaluate, variations, pset=pset, metadataset=loo_metadataset,
                                          surrogates=surrogates, subset=args.subset, toolbox=toolbox)
                variations = list(zip(fitnesses, [str(v) for v in variations]))
                best = max(variations)
                logging.info(str(best[1]))
            else:
                logging.info(str(ind))

        checks = problem.get('checks', [])
        for check_name, check_individual in checks.items():
            scale_result = list(toolbox.map(toolbox.evaluate, [
                creator.Individual(gp.PrimitiveTree.from_string(check_individual, pset))]))[0][0]
            logging.info("{}:{}".format(check_name, scale_result))


if __name__ == '__main__':
    main()
