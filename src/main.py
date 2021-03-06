import argparse
import datetime
import functools
import logging
import math
import os
import random
import time
import uuid
import pathlib
import platform
import sys

import numpy as np

from deap import tools

from evolution import setup_toolbox
from evolution.operations import mass_evaluate, mass_evaluate_2, n_primitives_in, \
    insert_fixed, approx_eq, try_compile_individual
from evolution.algorithms import one_plus_lambda, eaMuPlusLambda, random_search

from deap import gp, creator
from operator import attrgetter

from problem import Problem
from utils import str2bool


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
                        help="Maximum number of generations (default=300)",
                        dest='ngen', type=int, default=300)
    parser.add_argument('-a',
                        help="Algorithm. {mupluslambda, onepluslambda, random_search}",
                        dest='algorithm', type=str, default='mupluslambda')
    parser.add_argument('-esn',
                        help="Early Stopping N. Stop optimization if there is no improvement in n generations.",
                        dest='early_stop_n', type=int, default=20)
    parser.add_argument('-o',
                        help="Output directory."
                             "Write log, evaluations and progress to files in this dir.",
                        dest='output', type=str, default=None)
    parser.add_argument('-mno',
                        help="Max Number of Operators",
                        dest='max_number_operators', type=int, default=None)
    parser.add_argument('-oc',
                        help=(
                            "Optimize Constants. Instead of evaluating an individual with specific constants"
                            "evaluate based it on 50 random instantiation of constants instead."),
                        dest='optimize_constants', type=bool, default=False)
    parser.add_argument('-opt',
                        help="What to optimize for, mean or median.",
                        dest='aggregate', type=str, default='mean')
    parser.add_argument('-t',
                        help="Perform search and evaluation for this task only.",
                        dest='task', type=int, default=None)
    parser.add_argument('-k', '--leave-k-out',
                        help="Amount of tasks to use for out-of-sample evaluations,"
                             "if it is float 0<k<1 then that fraction will be left out.",
                        dest='leave_k_out', type=float, default=1)
    parser.add_argument('-f', '--fold',
                        help="The fold of k-fold validation to run, if None run all.",
                        dest='fold', type=int, default=None)
    parser.add_argument('-warm',
                        help=(
                            "Warm-start optimization by including the 'benchmark' solutions in the "
                            "initial population."),
                        dest='warm_start', type=str2bool, default=False)
    parser.add_argument('-s',
                        help="Evaluate individuals on a random [S]ubset of size [0, 1].",
                        dest='subset', type=float, default=1.)
    parser.add_argument('--seed',
                        # I am not sure why it does not fix other random decisions,
                        # but order for leave-k-out tasks is determined first so
                        # at that point the script doesn't diverge yet.
                        # I suspect floating point precision to be the culprit,
                        # but I have not looked into it.
                        help=("Seed for random number generator. "
                              "Currently only fixes samples for cross-validation"),
                        dest='seed', type=int, default=0)
    parser.add_argument('-cst',
                    help=("Search only constant formulas?"),
                    dest='constants_only', type=str2bool, default=False)
    parser.add_argument('-age',
                help=("Regularize age by killing of older population members every nth generation."
                      "Defaults to a 1e5 (every 1e5 generations)."),
                dest='age_regularization', type=float, default=1e5)
    parser.add_argument('-cxpb',
                help=("Probability a new candidate is generated by crossover,"
                      "otherwise it is generated through mutation."),
                dest='cxpb', type=float, default=0.25)
    parser.add_argument('-mss',
                help=("Set the Max Start Size: the maximum depth per subtree for each"
                      "hyperparameter. (default=3). Note: for random search, "
                      "all candidates are constrained this way."),
                dest='max_start_size', type=int, default=2)
    parser.add_argument('--description',
                help=("Description to log with the hyperparameter settings."
                      "May not contain a semicolon (';')."),
                dest='Description', type=str, default='-')
    return parser.parse_args()


def configure_logging(output_file: str = None):
    """ Configure INFO logging to console and optionally DEBUG to an output file. """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    if output_file is not None:
        log_file_handle = logging.FileHandler(output_file)
        log_file_handle.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(log_file_handle)


def calculate_folds(N, k):
    if k >= 1:
        n_folds = math.floor(N / k)
    else:
        n_folds = int(1 / k)

    fold_size = math.floor(N / n_folds)
    remainder = N % n_folds
    # we distribute one remainder per fold, starting with the first
    return [
        (i * fold_size + min(j, remainder), (i + 1) * fold_size + min(j, remainder) + (1 if remainder - j > 0 else 0))
        for j, i in enumerate(range(n_folds))
    ]


def main():
    run_id = str(uuid.uuid4())
    # Numpy must raise all warnings, otherwise overflows may go undetected.
    np.seterr(all='raise')
    args = cli_parser()
    if args.leave_k_out != 1 and args.task is not None:
        raise ValueError("It is not possible to use leave-k-out when specifically "
                         f"holding out a task (k={args.leave_k_out}, task={args.task})")

    test_task_random = random.Random(args.seed)

    if args.output:
        run_dir = os.path.join(args.output, run_id)
        pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
        log_file = os.path.join(run_dir, 'output.log')
        configure_logging(log_file)
        with open(os.path.join(run_dir, "evaluations.csv"), 'a') as fh:
            fh.write(f"run;fold;task;gen;inout;score;length;endresult;expression\n")
        with open(os.path.join(run_dir, "progress.csv"), 'a') as fh:
            fh.write(f"run;task;generation;score_min;score_avg;score_max\n")
        with open(os.path.join(run_dir, "finalpops.csv"), 'a') as fh:
            fh.write(f"run;fold;task;score;length;expression\n")
        with open(os.path.join(run_dir, "final_pareto.csv"), 'a') as fh:
            fh.write(f"run;fold;task;score;length;expression\n")
        with open(os.path.join(run_dir, "metadata.csv"), 'a') as fh:
            fh.write("field;value\n")
            for parameter, value in args._get_kwargs():
                fh.write(f"{parameter};{value}\n")
            fh.write(f"OS;{platform.system()} {platform.release()}\n")
            fh.write(f"Python;{platform.python_version()}\n")
            fh.write(f"start-date;{datetime.datetime.now().isoformat()}\n")

    else:
        configure_logging()

    time_start = time.time()

    problem = Problem(args.problem)

    logging.info(f"Starting problem: {args.problem}")
    for parameter, value in args._get_kwargs():
        logging.info(f"param:{parameter}:{value}")
    logging.info(f"runid:{run_id}")

    logging.info(f"Benchmark problems: {args.problem}")
    if not args.constants_only:
        for check_name, check_individual in problem.benchmarks.items():
            logging.info(f"{check_name} := {check_individual}")

    if (args.optimize_constants):
        mass_eval_fun = mass_evaluate_2
    else:
        mass_eval_fun = mass_evaluate

    # The 'toolbox' defines all operations, and the primitive set defines the grammar.
    toolbox, pset = setup_toolbox(problem, args)

    if args.algorithm == 'random_search':
        # iterations don't make much sense in random search,
        # so we modify the values to make better use of batch predictions.
        args.ngen = args.ngen // 100
        args.lambda_ = args.lambda_ * 100

    tasks = list(problem.metadata.index)
    if args.task is not None:
        if args.task not in tasks:
            raise ValueError(f"Requested task {args.task} not in metadata.")
        else:
            tasks = [args.task]

    if len(problem.fixed):
        logging.info(f"With fixed hyperparameters: {problem.fixed}:")
        logging.info(f"And hyperparameters: {problem.hyperparameters}:")

    # ================================================
    # Start evolutionary optimization
    # ================================================
    in_sample_mean = {}

    test_task_random.shuffle(tasks)
    if args.task:
        task_idx = tasks.index(args.task)
        folds = [(task_idx, task_idx + 1)]
    else:
        folds = calculate_folds(len(tasks), args.leave_k_out)

    for fold, (start, end) in enumerate(folds):
        if args.fold is not None and args.fold != fold:
            continue
        test_tasks = tasks[start:end]

        logging.info(f"START_TASKS: {fold}")
        # 'tasks' experiment data is used as validation set, so we must not use
        # it during our symbolic regression search.
        train_tasks = problem.metadata[~problem.metadata.index.isin(test_tasks)]

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
        hof = tools.ParetoFront(similar=approx_eq)
        last_best = (0, -10)
        last_best_gen = 0

        # Little hackery for logging with early stopping
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + mstats.fields
        early_stop_iter = args.ngen
        stop = False

        for i in range(args.ngen):
            # 'map' will be called within the optimization algorithm for batch evaluation.
            # All evaluation variables are fixed, except for the individuals themselves.
            toolbox.register(
                "map",
                functools.partial(
                    mass_eval_fun, pset=pset, metadataset=train_tasks,
                    surrogates=problem.surrogates, subset=1.0,
                    toolbox=toolbox, optimize_constants=args.optimize_constants,
                    problem=problem, agg=args.aggregate
                )
            )
            # Hacky way to integrate early stopping with DEAP.
            if args.algorithm == 'mupluslambda':
                if args.subset != 1.0 and i != args.ngen - 1:
                    hof = tools.ParetoFront(similar=approx_eq)
                    # 'map' will be called within the optimization algorithm for batch evaluation.
                    # All evaluation variables are fixed, except for the individuals themselves.
                    toolbox.register(
                        "map",
                        functools.partial(
                            mass_eval_fun, pset=pset, metadataset=train_tasks,
                            surrogates=problem.surrogates, subset=args.subset,
                            toolbox=toolbox, optimize_constants=args.optimize_constants,
                            problem=problem, agg=args.aggregate
                        )
                    )

                pop, _ = eaMuPlusLambda(
                    population=pop,
                    toolbox=toolbox,
                    mu=args.mu,  # Number of Individuals to pass between generations
                    lambda_=args.lambda_,  # Number of offspring per generation
                    cxpb=args.cxpb,
                    mutpb=1-args.cxpb,
                    ngen=1,
                    verbose=False,
                    halloffame=hof,
                    no_cache=(args.subset < 1.0),
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

            # ====================== MODIFICATION ========================
            # set current gen as birthyear for all newly initialized
            # individuals
            if i > 0 and args.output:
                with open(os.path.join(run_dir, "age.csv"), 'a') as fh:
                    birthyears = [i.birthyear for i in pop if i.birthyear is not None]
                    if len(birthyears) == 0:
                        # No assigned birthyear means all individuals are new.
                        fh.write(f"{i};0;0\n")
                    else:
                        fh.write(f"{i};{i - sum(birthyears)/ len(birthyears)};{max(i - b for b in birthyears)}\n")

            if i > args.age_regularization:
                # cull -all- individuals older than args.age_regularization
                old_pop_size = len(pop)
                pop_to_live = [ind for ind in pop if ind.birthyear is None or ind.birthyear >= (i - args.age_regularization)]
                killed_pop_size = old_pop_size - len(pop_to_live)
                pop = [*pop_to_live, *toolbox.population(n=killed_pop_size)]

            invalid_birthyear = [ind for ind in pop if ind.birthyear is None]
            for ind in invalid_birthyear:
                ind.birthyear = i

            if args.output:
                with open(os.path.join(run_dir, "progress.csv"), 'a') as fh:
                    fh.write(f"{run_id};{fold};{i};{record['fitness']['min']};{record['fitness']['avg']};{record['fitness']['max']}\n")
            logging.info(generation_info_string)

            stop = (i == args.ngen - 1)
            # Little hackery for logging early stopping
            for ind in hof:
                if ind.fitness.wvalues > last_best:
                    last_best = ind.fitness.wvalues
                    last_best_gen = i
            if i - last_best_gen > args.early_stop_n:
                early_stop_iter = min(i, early_stop_iter)
                if i == early_stop_iter and args.algorithm != "random_search":
                    stop = True

            # Evaluate in-sample and out-of-sample every N iterations OR
            # in the early stopping iteration
            if ((i > 1 and i % 50 == 0) or stop or args.algorithm == "random_search"):
                logging.info("Evaluating in sample:")
                for ind in sorted(hof, key=n_primitives_in):
                    scale_result, length = list(toolbox.map(toolbox.evaluate, [ind]))[0]
                    logging.info(f"[GEN_{i}|{ind}|{scale_result:.4f}]")
                    if args.output:
                        with open(os.path.join(run_dir, "evaluations.csv"), 'a') as fh:
                            fh.write(f"{run_id};{fold};0;{i};in;{scale_result:.4f};{length};{stop};{ind}\n")

                logging.info("Evaluating out-of-sample:")
                if args.output and stop:
                    save_to_final_file(test_tasks, pop, fold, problem, toolbox, pset, run_id,
                        filename=os.path.join(run_dir, "finalpops.csv")
                    )

                    save_to_final_file(test_tasks, hof, fold, problem, toolbox, pset, run_id,
                        filename=os.path.join(run_dir, "final_pareto.csv")
                    )

                for task in test_tasks:
                    for ind in sorted(hof, key=n_primitives_in):
                        score = get_surrogate_score(problem, task, ind, pset, toolbox)
                        logging.info(f"[GEN_{i}|{ind}|{score:.4f}]")
                        if args.output:
                            with open(os.path.join(run_dir, "evaluations.csv"), 'a') as fh:
                                fh.write(f"{run_id};{fold};{task};{i};test;{score:.4f};{n_primitives_in(ind)};{stop};{ind}\n")

            if stop:
                logging.info(f"Stopped early in iteration {early_stop_iter}, no improvement in {args.early_stop_n} gens.")
                break

        if not args.constants_only:
            logging.info("BENCHMARK: Evaluating in-sample:")
            in_sample_mean[task] = {}
            for check_name, check_individual in problem.benchmarks.items():
                individual = str_to_individual(check_individual, pset)
                scale_result, length = list(toolbox.map(toolbox.evaluate, [individual]))[0]
                logging.info(f"[{check_name}: {individual}|{scale_result:.4f}]")
                in_sample_mean[task][check_name] = scale_result
                if args.output:
                    with open(os.path.join(run_dir, "evaluations.csv"), 'a') as fh:
                        fh.write(f"{run_id};{fold};0;{i};in;{scale_result:.4f};{n_primitives_in(ind)};{True};{check_name}\n")

            logging.info("BENCHMARK:  Testing out-of-sample:")
            for check_name, check_individual in problem.benchmarks.items():
                for task in test_tasks:
                    ind = str_to_individual(check_individual, pset)
                    score = get_surrogate_score(problem, task, check_individual, pset, toolbox)
                    logging.info(f"[GEN_{i}|{task}|{check_name}: {ind}|{score:.4f}]")
                    if args.output:
                        with open(os.path.join(run_dir, "evaluations.csv"), 'a') as fh:
                            fh.write(f"{run_id};{fold};{task};{i};test;{score:.4f};{n_primitives_in(ind)};{True};{check_name}\n")
                            
    if args.output:
        with open(os.path.join(run_dir, "metadata.csv"), 'a') as fh:
            fh.write(f"end-date;{datetime.datetime.now().isoformat()}\n")


    # # Get benchmark scores across all tasks
    # for check_name, check_individual in problem.benchmarks.items():
    # logging.info(f"{check_name} := {check_individual}")
    # for check_name, check_individual in problem.benchmarks.items():
    # avg_val=np.mean([v[check_name] for k, v in in_sample_mean.items()])
    # logging.info("Average in_sample mean for {}: {}".format(check_name, avg_val))

    time_end = time.time()
    logging.info("Finished problem {} in {} seconds!".format(args.problem, round(time_end - time_start)))


def save_to_final_file(test_tasks, individuals, fold, problem, toolbox, pset, run_id, filename):
    with open(filename, 'a') as fh:
        for ind in individuals:
            fh.write(f"{run_id};{fold};0;{ind.fitness.wvalues[0]:.4f};{n_primitives_in(ind)};{ind}\n")
            for task in test_tasks:
                score = get_surrogate_score(problem, task, ind, pset, toolbox)
                fh.write(f"{run_id};{fold};{task};{score:.4f};{n_primitives_in(ind)};{ind}\n")


def str_to_individual(individual_str, pset):
    expression = gp.PrimitiveTree.from_string(individual_str, pset)
    individual = creator.Individual(expression)
    return individual


def get_surrogate_score(problem, task, individual, pset, toolbox) -> float:
    """ individual can be either an Individual or a str that represents one. """
    if isinstance(individual, str):
        individual = str_to_individual(individual, pset)

    fn_ = try_compile_individual(individual, pset, problem)
    mf_values = problem.metadata.loc[task]
    hp_values = toolbox.evaluate(fn_, mf_values)
    return problem.surrogates[task].predict(np.asarray(hp_values).reshape(1, -1))[0]


if __name__ == '__main__':
    main()
