""" Evaluates a given learner in a basic pipeline.

positional arguments:
    import_module: the module part from which the learner is imported, e.g. sklearn.ensemble
    create_call: the call which instantiates the learner with desired hyperparameters, can make us of task metafeatures.
                 E.g. RandomForestClassifier(n_estimators=100,max_features='sqrt',max_depth={p})
                 To use metafeatures, use their symbolic between brackets as above.
                 The following metafeatures are available:
                    - n: number of instances
                    - p: number of features
                    - mkd: median kernel distance
                    - rc: ratio of symbolic features to total number of features ([0,1])
                    - m: number of classes
                    - xvar: `X.var()`, variance of the data as a single array
                    - mcp: ratio of majority class ([0,1])
    output_file: file to which evaluation results should be written.

The pipeline in which the learner is evaluated is this:
    - impute: numeric values with mean, symbolic values with most frequent
    - transform: scale numeric values to zero mean, unit variance. one-hot encode symbolic values
    - select: remove any constant features
    - learn a model on the preprocessed data
"""

import sys
sys.path.append("D:\\repositories/openml-python")
sys.path.append("../../openml-python")

from math import sqrt
import openml
import pandas as pd
import numpy as np
import stopit

from sklearn.metrics import accuracy_score
from utils import simple_construct_pipeline_for_task


def main():
    import_module = sys.argv[1]
    exec('import {}'.format(import_module))

    create_call = sys.argv[2]  # e.g. RandomForest(n_estimators=100,max_features={mkd})
    call_parameters = [item.split('}')[0] for item in create_call.split('{')[1:]]

    output_file = sys.argv[3]
    with open(output_file, 'a') as fh:
        fh.write('task;f0;f1;f2;f3;f4;f5;f6;f7;f8;f9;avg;std\n')

    metadata = pd.read_csv('data/ppp_metadata.csv', index_col=0)
    runs = []
    for i, task_id in enumerate(metadata.sort_values(by='n').index):
        print('[{:2d}/100] starting task {}'.format(i, task_id))
        try:
            with stopit.ThreadingTimeout(seconds=3600) as cm:
                task = openml.tasks.get_task(task_id)

                param_dict = {metafeature: metadata.loc[task_id][metafeature] for metafeature in call_parameters}
                instantiate_call = '{}.{}'.format(
                    import_module,
                    create_call.format(**param_dict)
                )
                learner = eval(instantiate_call)

                pipeline = simple_construct_pipeline_for_task(task, learner)
                run = openml.runs.run_model_on_task(pipeline, task, upload_flow=False, avoid_duplicate_runs=False)
                scores = run.get_metric_fn(accuracy_score)

                with open(output_file, 'a') as fh:
                    scores_str = ';'.join([str(score) for score in scores])
                    fh.write('{};{};{};{}\n'.format(task_id, scores_str, np.mean(scores), np.std(scores)))
                runs.append(run)
        except Exception as e:
            print(str(e))


if __name__ == '__main__':
    main()
