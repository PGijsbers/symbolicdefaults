import sys
import itertools

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
    task = sys.argv[1]
    if len(sys.argv) > 2:
        problems = [sys.argv[2]]
    else:
        problems = ['knn', 'svm', 'glmnet', 'rf', 'rpart']  # , 'xgboost']
    if len(sys.argv) > 3:
        algorithms = [sys.argv[3]]  # random_search or mupluslambda
    else:
        algorithms = ["-a random_search", "-a mupluslambda", "-cst"]

    start_command = "python src/main.py mlr_{problem} -o $TMPDIR/{outdir} {alg} -t {task}"
    for problem, algorithm in itertools.product(problems, algorithms):
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
