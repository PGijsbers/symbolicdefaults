import sys
import itertools

job_header = """\
#!/bin/bash
#SBATCH -N 1
#SBATCH -t 36:00:00

module load 2019
module load Python/3.6.6-intel-2018b

cd ~/symbolicdefaults
source venv/bin/activate

"""

job_footer = """\

wait
"""

if __name__ == '__main__':
    if len(sys.argv) > 1:
        problems = [sys.argv[1]]
    else:
        problems = ['knn', 'svm', 'glmnet', 'rf', 'rpart', 'xgboost']
    if len(sys.argv) > 2:
        algorithms = [sys.argv[2]]  # random_search or mupluslambda
    else:
        algorithms = ["random_search", "mupluslambda"]

    start_command = "python src/main.py mlr_{problem} -o {log} -a {alg}"
    for problem, algorithm in itertools.product(problems, algorithms):
        job_name = f"jobs/{problem}_{algorithm}.job"
        with open(job_name, 'a') as fh:
            fh.write(job_header)

        for i in range(10):
            logfile = f"~/symbolicdefaults/runs/mlr_{problem}_{algorithm}_{i}.log"
            cmd = start_command.format(problem=problem, log=logfile, alg=algorithm)
            if algorithm == "random_search":
                cmd += ' -mss 3'
            with open(job_name, 'a') as fh:
                fh.write(cmd + ' &\n')

        with open(job_name, 'a') as fh:
            fh.write(job_footer)
