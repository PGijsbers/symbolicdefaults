from collections import namedtuple
import pandas as pd

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
        if line.startswith('INFO:root:START_TASK'):
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
        if line.startswith('INFO:root:GEN'):
            current_results['generations'].append(parse_generation_line(line))
        if line.startswith('INFO:root:make_tuple'):
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
