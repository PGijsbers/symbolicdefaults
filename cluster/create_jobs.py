"""
Script for generating jobs.

Expects positional arguments:
1: either a task id or a csv file with a column of tasks
2: a problem to generate jobs for ['knn', 'svm', 'glmnet', 'rf', 'rpart']
   if no problem is specified, jobs are created for all problems.
3: an algorithm to find the default with ["-a random_search", "-a mupluslambda", "-cst"]
   if no algorithm is specified, jobs are created for all algorithms.
"""
import argparse
import itertools
import os
import sys
import uuid

sys.path.append("./src/")

import pandas as pd

from problem import Problem


job_header = """\
#!/bin/bash
#SBATCH -N 1
#SBATCH -t 00:30:00

module load 2019
module load Python/3.6.6-intel-2019b

cd ~/symbolicdefaults
source venv/bin/activate

"""

job_footer = """\

wait
"""


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def cli_parser():
    parser = argparse.ArgumentParser(description="Queue jobs for Symbolic Defaults experiments.")
    parser.add_argument('-p', type=str, default='all', dest="problems",
                        help="Problem(s) to optimize separated by a comma. (default=all)")
    parser.add_argument('-t', type=str, default='all', dest="tasks",
                        help="Task(s) for which to perform hold-one-out (default=all).")
    parser.add_argument('-a', type=str, default='mupluslambda', dest="algorithm",
                        help="Algorithm for optimization [mupluslambda*, random_search].")
    parser.add_argument('-age', type=int, default=None, dest='age')
    parser.add_argument('-s', type=float, default=None, dest='subset') 
    parser.add_argument('-ngen',
                        help="Maximum number of generations (default=300)",
                        dest='ngen', type=int, default=300)
    parser.add_argument('-esn',
                        help="Early Stopping N. Stop optimization if there is no improvement in n generations.",
                        dest='early_stop_n', type=int, default=20)
    parser.add_argument('-queue',
                        help='If set, automatically queue the job.',
                        dest='queue', type=str2bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = cli_parser()
    if args.problems == "all":
        problems = ['svm', 'knn', 'glmnet', 'rf', 'rpart']  # , 'xgboost']
    else:
        problems = args.problems.split(',')
    tasks = args.tasks.split(',')

    start_command = "python src/main.py mlr_{problem} -o $TMPDIR/{outdir} -a {alg} -t {task}"
    for problem in problems:
        problem_data = Problem(f"mlr_{problem}")
        if args.tasks == 'all':
            tasks = list(problem_data.metadata.index)
        for task in tasks:
            job_name = f"jobs/{problem}_{args.algorithm}_{task}_{str(uuid.uuid4())}.job"
            outdir = f"results/{problem}_{args.algorithm}/"
            with open(job_name, 'a') as fh:
                fh.write(job_header)

            # xgboost surrogates are huge, and during loading peak memory usage may be as high as 25Gb
            # To avoid MemoryErrors, we must run fewer tasks in parallel, and avoid loading all at once.
            if problem == "xgboost":
               n_jobs = 5
            else:
               n_jobs = 10
               delay = ''

            for i in range(n_jobs):
                search = start_command.format(problem=problem, outdir=outdir, alg=args.algorithm, task=task)
                if args.algorithm == "random_search":
                    search += ' -mss 3'
                if problem == "xgboost":
                    delay = f"sleep {(i // 2)*90};"
                if args.ngen != 300:
                    search += f' -ngen {args.ngen}'
                if args.early_stop_n != 20:
                    search += f' -esn {args.early_stop_n}'
                if args.subset is not None:
                    search += f' -s {args.subset}'
                if args.age is not None:
                    search += f' -age {args.age}'

                mkdir = f"mkdir -p ~/results"
                move = f"cp -r $TMPDIR/{outdir} ~/results"
                with open(job_name, 'a') as fh:
                    fh.write(f"({delay}{search};{mkdir};{move}) &\n")

            with open(job_name, 'a') as fh:
                fh.write(job_footer)

            if args.queue:
                os.system("sbatch %s" % job_name)

            with open("start_jobs.sh", newline='\n', mode='a') as fh:
                fh.write(f"sbatch {job_name}\n")
