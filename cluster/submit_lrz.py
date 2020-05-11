#!/usr/bin/env python

import os

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


job_directory = f"{os.getcwd()}/.job"
mkdir_p(job_directory)

def mkoutstring(job, search_method, suffix, constants_only, moreargs):
 if constants_only:
     moreargs = moreargs+"_cst"
 return(f"{job}_{search_method}_{moreargs}_{suffix}")

def runjob(job, search_method, constants_only=False, suffix="lrz", moreargs=""):
    '''Define a slurm job for searching symbolic defaults'''

    # Cluster configs; hardcoded for now as this will most likely be constant across partitions
    cluster="serial"           # Submit to serial cluster
    partition="serial_std"     # Submit to standard partition
    mem = 11000                # 22 GB Memory limit
    hrs = 12                   # 12 hours walltime

    # xgboost needs more memory
    if job == "mlr_xgboost":
        mem=22000

    outfile = mkoutstring(job, search_method, suffix, constants_only, moreargs)
    job_file = os.path.join(job_directory, f"{job}.job")
    logfile = f"runs/{outfile}.log"

    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name={job}.job\n")
        fh.writelines(f"#SBATCH --output=.out/{outfile}.out\n")
        fh.writelines(f"#SBATCH --error=.out/{outfile}.err\n")
        fh.writelines(f"#SBATCH --time={hrs}:00:00\n")
        fh.writelines(f"#SBATCH --mem={mem}mb\n")
        fh.writelines(f"#SBATCH --clusters={cluster}\n")
        fh.writelines(f"#SBATCH --partition={partition}\n")
        fh.writelines("module load slurm_setup\n")
        fh.writelines("module load spack\n")
        fh.writelines("module load python/3.6_intel\n")
        fh.writelines(f"python3.6 src/main.py {job} -a={search_method} -cst={constants_only} {moreargs} -o={logfile} -emut gaussian -ephs one -cx d1\n")

    os.system("sbatch %s" %job_file)

# Sumit a job for all combinations of job/search_method/cst
jobs=            ["mlr_svm", "mlr_glmnet", "mlr_knn", "mlr_rf", "mlr_rpart", "mlr_xgboost"] #, "svc_rbf", "adaboost"]
search_methods=  ["mupluslambda"] #["random_search", "mupluslambda"]
csts=            [False] #[True, False]

for job in jobs:
    for sm in search_methods:
        for cst in csts:
            for rep in range(3):
                runjob(job, sm, constants_only=cst, suffix=f"lrz_{rep}")
