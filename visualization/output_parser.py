import sys
sys.path.append('../')
from persistence import load_results_for_problem, load_problem

from collections import namedtuple
import pandas as pd
import numpy as np

gen_info = namedtuple("GenerationInformation",
                      field_names=[
                          'generation',
                          'score_min',
                          'score_avg',
                          'score_max',
                          'size_min',
                          'size_avg',
                          'size_max',
                      ])

optimization_result = namedtuple("OptimizationResult",
                                 field_names=[
                                     "task",
                                     "n_generations_elapsed",
                                     "stopped_early",
                                     "information_by_generation",
                                     "top_five_expressions"
                                 ])


def parse_generation_line(line: str) -> 'GenerationInformation':
    """ Parses a line with information about a generation's performance. """
    # INFO:root:GEN_0_FIT_0.6199841029394941_0.7400412770795781_0.8680925868543413_SIZE_1_2.4_7
    _, gen, _, min_fit, avg_fit, max_fit, _, min_size, avg_size, max_size = \
        line[:-1].split('_')
    return gen_info(gen, min_fit, avg_fit, max_fit, min_size, avg_size, max_size)


def parse_eo_console_output(file):
    with open(file, 'r') as fh:
        lines = fh.readlines()

    results = []
    current_results = dict(task=-1, generations=[], expressions=[])
    for line in lines:
        if line.startswith('START_TASK'):
            if current_results['task'] != -1:
                results.append(optimization_result(
                    task=current_results['task'],
                    n_generations_elapsed=len(current_results['generations']),
                    stopped_early=len(current_results['generations']) < 100,
                    information_by_generation=current_results['generations'],
                    top_five_expressions=current_results['expressions']
                ))

            current_results = dict(task=-1, generations=[], expressions=[])
            current_results['task'] = line.split(':')[-1][:-1]
        if line.startswith('GEN'):
            current_results['generations'].append(parse_generation_line(line))
        if line.startswith('make_tuple'):
            current_results['expressions'].append(line.split('(', 1)[1].rsplit(')', 1)[-2])

    return results


def is_performance_line(line):
    return line.count(' ') == 2


def parse_performance_line(line):
    task, avg, std = line[:-1].split(' ')
    return int(task), float(avg), float(std)


def get_performance_from_console_output(file):
    with open(file, 'r') as fh:
        lines = fh.readlines()
    performance_lines = [parse_performance_line(line) for line in lines
                         if is_performance_line(line)]
    return pd.DataFrame(performance_lines, columns=['task', 'avg', 'std']).set_index('task')


def get_performance_from_csv(file):
    # 125921;0.72;0.78;0.7;0.72;0.8;0.66;0.68;0.68;0.6;0.74;0.708;0.05528109984434102
    # task;fold 1;...fold 10;mean;std
    return pd.read_csv(file, index_col=0, sep=';')


def load_random_search_results(problem_name):
    p = load_problem(problem_name)
    return load_results_for_problem(p)


def generate_comparisons(problem, files, names):
    # The grid search result from Jan.
    svc_results = load_random_search_results(problem)

    # results currently still stored in log. should be aggregated to single file..
    performances = {name: None for name in names}
    for name, file in zip(names, files):
        performances[name] = get_performance_from_csv(file)

    df = pd.DataFrame(np.zeros(shape=(len(names), len(names) + 2)), columns=names + ['loss', 'N'])
    df.index = names

    # Calculate 'wins'
    for (method, performance) in performances.items():
        for (method2, performance2) in performances.items():
            one_over_two = (performance.avg - performance2.avg) > 0
            df.loc[method][method2] = sum(one_over_two)

    # Calculate loss
    for method, performance in performances.items():
        loss_sum = 0
        for i, row in performance.iterrows():
            if row.name in svc_results.task_id.unique():
                best_score = svc_results[svc_results.task_id == row.name].predictive_accuracy.max()
                loss = best_score - row.avg
                if loss < 0:
                    print('{} outperformed best on task {} by {}'.format(method, row.name, loss))
                loss_sum += loss
        df.loc[method]['loss'] = loss_sum
        df.loc[method]['N'] = len(performance)

    return df
