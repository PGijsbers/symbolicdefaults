import sys

job_template = """\
#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00

#cp -r ~/.openml/cache/ "$TMPDIR"/oml

module load python/3.5.0
python3 main.py {}
"""

if __name__ == '__main__':
    job_name = sys.argv[1]

    with open(job_name, 'w', newline='\n') as fh:
        fh.write(job_template.format(' '.join(["'{}'".format(el) for el in sys.argv[2:]])))
