import argparse
import functools
import logging
import sys

import numpy as np
import pandas as pd

from deap import tools

from evolution import setup_toolbox
from evolution.operations import mass_evaluate, mass_evaluate_2, n_primitives_in, mut_all_constants, insert_fixed
from evolution.algorithms import one_plus_lambda, eaMuPlusLambda, random_search

from deap import gp, creator
from operator import attrgetter

from problem import Problem


def cli_parser():
    description = "Use Symbolic Regression to find symbolic hyperparameter defaults."
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
                        dest='ngen', type=int, default=200)
    parser.add_argument('-a',
                        help="Algorithm. {mupluslambda, onepluslambda}",
                        dest='algorithm', type=str, default='mupluslambda')
    parser.add_argument('-s',
                        help="Evaluate individuals on a random [S]ubset of size [0, 1].",
                        dest='subset', type=float, default=1.)
    parser.add_argument('-esn',
                        help="Early Stopping N. Stop optimization if there is no improvement in n generations.",
                        dest='early_stop_n', type=int, default=20)
    parser.add_argument('-o',
                        help="Output file. Also write log output to this file.",
                        dest='output_file', type=str, default=None)
    parser.add_argument('-mno',
                        help="Max Number of Operators",
                        dest='max_number_operators', type=int, default=None)
    parser.add_argument('-oc',
                        help=(
                            "Optimize Constants. Instead of evaluating an individual with specific constants"
                            "evaluate based it on 50 random instantiation of constants instead."),
                        dest='optimize_constants', type=bool, default=False)
    parser.add_argument('-t',
                        help="Perform search and evaluation for this task only.",
                        dest='task', type=int, default=None)
    parser.add_argument('-warm',
                        help=(
                            "Warm-start optimization by including the 'benchmark' solutions in the "
                            "initial population."),
                        dest='warm_start', type=bool, default=False)
    parser.add_argument('-cst',
                    help=("Search only constant formulas?"),
                    dest='constants_only', type=bool, default=False)
    parser.add_argument('-age',
                help=("Regularize age by killing of older population members every nth generation."
                      "Defaults to a 1e5 (every 1e5 generations)."),
                dest='age_regularization', type=float, default=1e5)
    return parser.parse_args()


def configure_logging(output_file: str = None):
    """ Configure INFO logging to console and optionally DEBUG to an output file. """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    if output_file is not None:
        log_file_handle = logging.FileHandler(output_file)
        log_file_handle.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(log_file_handle)


def main():
    # Numpy must raise all warnings, otherwise overflows may go undetected.
    np.seterr(all='raise')

    args = cli_parser()
    configure_logging(args.output_file)
    problem = Problem(args.problem)

    for parameter, value in args._get_kwargs():
        logging.info(f"param:{parameter}:{value}")

    if (args.optimize_constants):
        mass_eval_fun = mass_evaluate_2
    else:
        mass_eval_fun = mass_evaluate

    # The 'toolbox' defines all operations, and the primitive set defines the grammar.
    toolbox, pset = setup_toolbox(problem, args)

    # ================================================
    # Start evolutionary optimization
    # ================================================
    top_5s = {}
    in_sample_mean = {}

    tasks = list(problem.metadata.index)
    if args.task is not None:
        if args.task not in tasks:
            logging.error("Requested task not in metadata.")
            quit(-1)
        else:
            tasks = [args.task]
    
    if len(problem.fixed):
        logging.info(f"With fixed hyperparameters: {problem.fixed}:")
        logging.info(f"And hyperparameters: {problem.hyperparameters}:")
    for task in tasks:
        logging.info(f"START_TASK: {task}")
        # 'task' experiment data is used as validation set, so we must not use
        # it during our symbolic regression search.
        loo_metadataset = problem.metadata[problem.metadata.index != task]

        # 'map' will be called within the optimization algorithm for batch evaluation.
        # All evaluation variables are fixed, except for the individuals themselves.
        toolbox.register(
            "map",
            functools.partial(
                mass_eval_fun, pset=pset, metadataset=loo_metadataset,
                surrogates=problem.surrogates, subset=args.subset,
                toolbox=toolbox, optimize_constants=args.optimize_constants,
                problem=problem
            )
        )

        # Seed population with configurations from problem.benchmark // fully "Symc" config
        pop = []
        if args.optimize_constants:
            pop = [*pop, *toolbox.population_symc(problem)]
        if args.warm_start:
            pop = [*pop, *toolbox.population_benchmark(problem)]
        pop = [*pop, *toolbox.population(n=max(0, args.lambda_ - len(pop)))]

        P = pop[0]
        
        # Set up things to track on the optimization process
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_size = tools.Statistics(n_primitives_in)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        hof = tools.ParetoFront()
        last_best = (0, -10)
        last_best_gen = 0

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
                    no_cache=(args.subset < 1.0)
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

            # ====================== MODIFICATION ========================
            # Kill off the oldest in the population
            if i > 1:
                # set current gen as birthyear for all newly initialized
                # individuals
                invalid_birthyear = [ind for ind in pop if not ind.birthyear]
                for ind in invalid_birthyear:
                    ind.birthyear = i
                if not i % args.age_regularization:
                    # delete the oldest (sort by birthyear) individual
                    pop = sorted(pop, key=attrgetter("birthyear"))[1:]
            # ============================================================

            # Little hackery for logging with early stopping
            record = mstats.compile(pop) if mstats is not None else {}
            logbook.record(gen=i, nevals=100, **record)

            generation_info_string = (
                "GEN_{}_FIT_{}_{}_{}_SIZE_{}_{}_{}".format(
                    i, record['fitness']['min'], record['fitness']['avg'],
                    record['fitness']['max'], record['size']['min'],
                    record['size']['avg'], record['size']['max']
                )
            )

            logging.info(generation_info_string)
            # Little hackery for logging early stopping
            for ind in hof:
                if ind.fitness.wvalues > last_best:
                    last_best = ind.fitness.wvalues
                    last_best_gen = i
            if i - last_best_gen > args.early_stop_n:
                logging.info(f"Stop early, no improvement in {args.early_stop_n} gens.")
                break

        # logging.info(f"Top 5 for task {task}:")
        # for ind in sorted(hof[:5], key=n_primitives_in):
        #     if args.optimize_constants:
        #         # since 'optimization' of constants is not saved,
        #         # reoptimize constants before printing.
        #         variations = [mut_all_constants(toolbox.clone(ind), pset)
        #                       for _ in range(50)]
        #         fitnesses = mass_evaluate(
        #             toolbox.evaluate, variations, pset=pset,
        #             metadataset=loo_metadataset, surrogates=problem.surrogates,
        #             subset=args.subset, toolbox=toolbox
        #         )
        #         variations = list(zip(fitnesses, [str(v) for v in variations]))
        #         best = max(variations)
        #         logging.info(str(best[1]))
        #     else:
        #         logging.info(str(ind))

        logging.info("Evaluating in sample:")
        for ind in sorted(hof, key=n_primitives_in):
            scale_result = list(toolbox.map(toolbox.evaluate, [ind]))[0][0]
            logging.info(f"[{ind}|{scale_result:.4f}]")


        
        if not args.constants_only:
            in_sample_mean[task] = {}
            for check_name, check_individual in problem.benchmarks.items():
                expression = gp.PrimitiveTree.from_string(check_individual, pset)
                individual = creator.Individual(expression)
                scale_result = list(toolbox.map(toolbox.evaluate, [individual]))[0][0]
                logging.info(f"[{check_name}: {individual}|{scale_result:.4f}]")
                in_sample_mean[task][check_name] = scale_result

        logging.info("Evaluating out-of-sample:")
        for ind in sorted(hof, key=n_primitives_in):
            fn_ = gp.compile(ind, pset)
            mf_values = problem.metadata.loc[task]
            hp_values = insert_fixed(toolbox.evaluate(fn_, mf_values), problem)
            score = problem.surrogates[task].predict(np.asarray(hp_values).reshape(1, -1))
            logging.info(f"[{ind}|{score[0]:.4f}]")

        if not args.constants_only:  
            for check_name, check_individual in problem.benchmarks.items():
                expression = gp.PrimitiveTree.from_string(check_individual, pset)
                ind = creator.Individual(expression)
                fn_ = gp.compile(ind, pset)
                mf_values = problem.metadata.loc[task]
                hp_values = toolbox.evaluate(fn_, mf_values)
                score = problem.surrogates[task].predict(np.asarray(hp_values).reshape(1, -1))
                logging.info(f"[{check_name}: {ind}|{score[0]:.4f}]")

    if not args.constants_only:
        for check_name, check_individual in problem.benchmarks.items():
            logging.info(f"{check_name} := {check_individual}")

        for check_name, check_individual in problem.benchmarks.items():
            avg_val=np.mean([v[check_name] for k, v in in_sample_mean.items()])
            logging.info("Average in_sample mean for {}: {}".format(check_name, avg_val))

if __name__ == '__main__':
    main()
