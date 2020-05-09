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
 return(f"runs/{job}_{search_method}_{moreargs}_{suffix}.log")

def runjob(job, search_method, constants_only=False, suffix="lrz", moreargs=""):
    '''Define a slurm job for searching symbolic defaults'''

    # Cluster configs; hardcoded for now as this will most likely be constant across partitions
    cluster="serial"           # Submit to serial cluster
    partition="serial_std"     # Submit to standard partition
    mem = 12000                # 12 GB Memory limit
    hrs = 12                   # 12 hours walltime

    outfile = mkoutstring(job, search_method, suffix, constants_only, moreargs)
    job_file = os.path.join(job_directory, f"{job}.job")
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name={job}.job\n")
        fh.writelines(f"#SBATCH --output=.out/{job}.out\n")
        fh.writelines(f"#SBATCH --error=.out/{job}.err\n")
        fh.writelines(f"#SBATCH --time={hrs}:00:00\n")
        fh.writelines(f"#SBATCH --mem={mem}\n")
        fh.writelines(f"#SBATCH --clusters={cluster}\n")
        fh.writelines(f"#SBATCH --partition={partition}\n")
        fh.writelines("module load slurm_setup\n")
        fh.writelines("module load spack\n")
        fh.writelines("module load python/3.6_intel\n")
        fh.writelines(f"python3 src/main.py {job} -a={search_method} -cst={constants_only} {moreargs} -o={outfile}\n")

    os.system("sbatch %s" %job_file)

jobs=["mlr_svm", "mlr_glmnet", "mlr_knn", "mlr_rf", "mlr_rpart", "mlr_xgboost", "svc_rbf", "adaboost"]
search_method=["random_search", "mupluslambda"]

for job in jobs[1:2]:
    runjob(job, "mupluslambda", moreargs="-age=3")
