from typing import Tuple
from collections import defaultdict
import numpy as np
import pandas as pd


def parse_log(file, with_prefix=False, benchmarks=None):
    with open(file) as fh:
        lines = fh.readlines()

    p = 'INFO:root:' if with_prefix else ''

    definitions = [i for i, line in enumerate(lines) if ':=' in line]
    first_definition = definitions[0] if definitions else -1
    benchmarks = benchmarks if benchmarks else set()

    print("The predefined defaults are:")
    for line in lines[first_definition:]:
        if ':=' in line:
            print(f" * {line[len(p):-1]}")
            benchmarks.add(line[len(p):].split(' :=')[0])

    task_starts = [i for i, line in enumerate(lines) if "START_TASK:" in line]
    in_sample_starts = [i for i, line in enumerate(lines) if "Evaluating in sample:" in line]
    out_sample_starts = [i for i, line in enumerate(lines) if "Evaluating out-of-sample:" in line]

    def parse_evaluation_line(line) -> Tuple[str, int, float]:
        """ Parse an evaluation line, returning the expression or name, its 'length' and the score.

        e.g. INFO:root:[make_tuple(p, mkd)|0.8893]\n -> 'make_tuple(p, mkd)', 1, 0.8893
        Length is 0 for benchmark problems.
        """
        start, pipe, end = line.find('['), line.find('|'), line.find(']')
        expression = line[start + 1: pipe]
        expression_length = expression.count('(')
        return expression, expression_length, float(line[pipe + 1: end])

    tasks = [int(line[:-1].split(": ")[-1]) for line in lines if "START_TASK:" in line]
    idx = pd.MultiIndex.from_product([tasks, ["in-sample", "out-sample"]], names=['task', 'sample-type'])
    df = pd.DataFrame(index=idx, columns=["score-1","score-2", "score-3", "score-best", *benchmarks], dtype=float)

    expressions_by_length = defaultdict(list)
    generations_by_task = {}

    for task_start, in_start, out_start, next_task in zip(task_starts, in_sample_starts,
                                                          out_sample_starts,
                                                          task_starts[1:] + [
                                                              first_definition]):
        # start line looks like: INFO:root:START_TASK: 29\n
        task = int(lines[task_start][:-1].split(": ")[-1])

        # Since the in-sample evaluation message follows directly after optimization is done, we use that to record
        # the number of generations. We account for the early stopping message if it did not run to 200 generations.
        ended_early = 0 if in_start - task_start == 201 else - 1
        generations_by_task[task] = in_start - (task_start + 1) - ended_early

        # Following the "INFO:root:Evaluating in sample:" message, symbolic default performance are printed
        # They are formatted as "INFO:root:[make_tuple(p, mkd)|0.8893]"
        # First is any number of best solutions from the pareto front. The last four are benchmark solutions.
        # It is possible that two equally good solutions are printed (i.e. same length and performance).
        expr_in_task = set()
        max_length = 0

        for in_sample_evaluation in lines[in_start + 1: out_start]:
            expr, length, score = parse_evaluation_line(in_sample_evaluation)
            # Pareto fronts may contain literal duplicates, so we filter those out manually.
            if expr not in expr_in_task:
                expressions_by_length[length].append(expr)
                expr_in_task.add(expr)

            if length != 0:
                if length < 4:
                    # Only report one out-of-sample solution for each length (and all benchmarks), so overwrite is OK.
                    df.loc[task, "in-sample"][f"score-{length}"] = score

                # Update best so far score and maximum length
                df.loc[task, "in-sample"][f"score-best"] = np.nanmax(
                    [score, df.loc[task, "in-sample"][f"score-best"]])
                max_length = max(max_length, length)
            else:
                df.loc[task, "in-sample"][expr] = score

            if length > max_length:
                max_length = length  # To know for which length "best" should score out of sample

        # Because two equal solutions can be in the Pareto front,
        # we note the average out of sample performance if multiple solutions were found.
        # Naturally, the solutions with the best in-sample score were those with the highest length in the Pareto front.

        scores_by_length = defaultdict(list)

        for out_sample_evaluation in lines[out_start + 1: next_task]:
            expr, length, score = parse_evaluation_line(out_sample_evaluation)
            if length != 0:
                scores_by_length[length].append(score)
            else:
                df.loc[task, "out-sample"][expr] = score

        for length, scores in scores_by_length.items():
            if length < 4:
                df.loc[task, "out-sample"][f"score-{length}"] = np.mean(scores)
            if length == max_length:
                df.loc[task, "out-sample"][f"score-best"] = np.mean(scores)
            if np.mean(scores) == float("nan"):
                print('hi')

    return df, expressions_by_length, generations_by_task


def comparisons(df):
    out_sample = df.index.map(lambda idx: idx[1] == "out-sample")

    alone = {k: 0 for k in df.iloc[0].index.values}
    shared = {k: 0 for k in df.iloc[0].index.values}

    for _, out in df.loc[out_sample].iterrows():
        best = out[out == out.max()].index.values
        if len(best) == 1:
            alone[best[0]] += 1
        else:
            for winner in best:
                shared[winner] += 1

    alone = {k: alone[k] for k in sorted(alone)}
    shared = {k: shared[k] for k in sorted(shared)}
    either = {k: shared[k] + alone[k] for k in sorted({**alone, **shared})}
    comparison = pd.DataFrame([alone, shared, either], index=['alone', 'shared', 'either'])

    df_out = df.loc[out_sample].copy()
    df_out['max'] = df_out.max(axis=1)
    for col in df_out:
        df_out['d_' + col] = df_out['max'] - df_out[col]
    d_cols = [c for c in df_out.columns if c.startswith('d_')]
    df_out[d_cols].mean()
    df_out[d_cols].median()

    in_sample = df.index.map(lambda idx: idx[1] == "in-sample")
    df.loc[in_sample].idxmax(axis=1).value_counts()
    df.loc[in_sample][reversed(df.columns)].idxmax(axis=1).value_counts()
    return comparison, df_out


if __name__ == '__main__':
    quit()
    # for length, expressions in sorted(expressions_by_length.items()):
    #     m = max(set(expressions), key=expressions.count)
    #     print(
    #         f" Found {len(expressions):3d} expressions of length {length}. Most frequent: {m} ({expressions.count(m)} times)")
    # pd.Series(generations_by_task, name="generations").hist(bins=20)
    # pd.Series(generations_by_task, name="generations").median()
