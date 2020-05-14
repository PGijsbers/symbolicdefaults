"""
Script for generating jobs.

Expects positional arguments:
1: either a task id or a csv file with a column of tasks
2: a problem to generate jobs for ['knn', 'svm', 'glmnet', 'rf', 'rpart']
   if no problem is specified, jobs are created for all problems.
3: an algorithm to find the default with ["-a random_search", "-a mupluslambda", "-cst"]
   if no algorithm is specified, jobs are created for all algorithms.
"""

import os
import sys
import itertools
import pandas as pd

job_header = """\
#!/bin/bash
#SBATCH -N 1
#SBATCH -t 1:00:00

module load 2019
module load Python/3.6.6-intel-2018b

cd ~/symbolicdefaults
source venv/bin/activate

"""

job_footer = """\

wait
"""

if __name__ == '__main__':
    if os.path.exists(sys.argv[1]):
        df = pd.read_csv(sys.argv[1], index_col=0)
        tasks = df.index.values
    else:
        tasks = sys.argv[1]
    if len(sys.argv) > 2:
        problems = [sys.argv[2]]
    else:
        problems = ['knn', 'svm', 'glmnet', 'rf', 'rpart']  # , 'xgboost']
    if len(sys.argv) > 3:
        algorithms = [sys.argv[3]]  # random_search or mupluslambda
    else:
        algorithms = ["-a random_search", "-a mupluslambda", "-cst"]

    start_command = "python src/main.py mlr_{problem} -o $TMPDIR/{outdir} {alg} -t {task}"
    for problem, algorithm, task in itertools.product(problems, algorithms, tasks):
        alg_short = algorithm.split(' ')[-1]
        job_name = f"jobs/{problem}_{alg_short}_{task}.job"
        with open(job_name, 'a') as fh:
            fh.write(job_header)

        for i in range(10):
            search = start_command.format(problem=problem, outdir='results', alg=algorithm, task=task)
            if algorithm == "random_search":
                search += ' -mss 3'
            move = "cp -r $TMPDIR/{outdir} ~/{outdir}"
            with open(job_name, 'a') as fh:
                fh.write(f"({search}; {move}) &\n")

        with open(job_name, 'a') as fh:
            fh.write(job_footer)

        with open("start_jobs.sh", newline='\n', mode='a') as fh:
            fh.write(f"sbatch {job_name}\n")
