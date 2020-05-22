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
#SBATCH -t 0:15:00

module load 2019
module load Python/3.6.6-intel-2018b

cd ~/symbolicdefaults
source venv/bin/activate

"""

job_footer = """\

wait
"""

if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])
    combinations = [[(search, task, problem)] * repeat for _, (search, task, repeat, problem) in df.iterrows()]
    combinations = list(itertools.chain(*combinations))

    start_command = "python src/main.py mlr_{problem} -o $TMPDIR/{outdir} {alg} -t {task}"
    for i in range(0, len(combinations) // 5):
        job_name = f"jobs/missing_{i}.job"
        with open(job_name, 'a') as fh:
            fh.write(job_header)

        for j, (search, task, problem) in enumerate(combinations[i*5:(i+1)*5]):
            outdir = f"results/{problem}_{search}/"
            if search == "random_search":
                algorithm = '-a random_search -mss 3'
            elif search == "True":
                algorithm = '-a mupluslambda -cst True'
            else:
                algorithm = '-a mupluslambda'
            cmd = start_command.format(problem=problem, outdir=outdir, alg=algorithm, task=int(float(task)))

            if problem == "xgboost":
                delay = f"sleep {(j // 2)*90};"
            else:
                delay = ''

            mkdir = f"mkdir -p ~/results"
            move = f"cp -r $TMPDIR/{outdir} ~/results"
            with open(job_name, 'a') as fh:
                fh.write(f"({delay}{cmd};{mkdir};{move}) &\n")

        with open(job_name, 'a') as fh:
            fh.write(job_footer)

        with open("start_jobs.sh", newline='\n', mode='a') as fh:
            fh.write(f"sbatch {job_name}\n")
