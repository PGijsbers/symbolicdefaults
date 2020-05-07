#!/usr/bin/env python

import os

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


job_directory = f"{os.getcwd()}/.job"
mkdir_p(job_directory)


cluster="serial"
partition="serial_mpp2"
mem = 12000
hrs = 12

suffix = "_lrz"

jobs=["mlr_svm", "mlr_glmnet"]

for job in jobs:
    job_file = os.path.join(job_directory, f"{job}.job")
    with open(job_file) as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name={job}.job\n")
        fh.writelines(f"#SBATCH --output=.out/{job}.out\n")
        fh.writelines(f"#SBATCH --error=.out/{job}.err\n")
        fh.writelines(f"#SBATCH --time={hrs}-00:00\n")
        fh.writelines(f"#SBATCH --mem={mem}\n" % mem)
        fh.writelines(f"#SBATCH --clusters={cluster}")
        fh.writelines(f"#SBATCH --mem_per_cpu={mem}\n" % mem)
        fh.writelines(f"#SBATCH --clusters={cluster}")
        fh.writelines(f"#SBATCH --partition={partition}\n")
        fh.writelines(f"python3 src/main.py {job} -o=runs/{job}_{suffix}.log")

    os.system("sbatch %s" %job_file)
