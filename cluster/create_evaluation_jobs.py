import sys

job_template = """\
#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00

cp -r ~/.openml/cache/ "$TMPDIR"/oml

module load python/3.5.0
python3 evaluate_pipeline.py {}
"""

if __name__ == '__main__':
    job_name = sys.argv[1]
    import_module = sys.argv[2]  # e.g. sklearn.ensemble
    create_call = sys.argv[3]  # e.g. RandomForest(n_estimators=10,max_features={mkd})

    with open(job_name, 'w', newline='\n') as fh:
        fh.write(job_template.format(' '.join(["'{}'".format(el) for el in sys.argv[2:]])))
