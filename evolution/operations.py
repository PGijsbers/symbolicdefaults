import functools
import typing

import numpy as np
import pandas as pd
from deap import gp


def n_primitives_in(individual):
    """ Return the number of primitives in the individual. """
    return len([e for e in individual if isinstance(e, gp.Primitive)])


def try_evaluate_function(fn, input_, invalid):
    """ Return fn(input_) if output is sequence of finite float32, else return `invalid`.

    This function is used to evaluate the symbolic expressions for a particular dataset.

    :param fn: callable. Takes input `input_` and should return a sequence of numbers.
    :param input_: input arguments to `fn`
    :param invalid: return value if `fn(input_)` does not return a sequence of finite float32 values.
    :return:
        `fn(input_)` if its output is a sequence of finite float32 values, else `invalid`
    """
    try:
        values = fn(**dict(input_))
        if not all([np.isfinite(np.float32(val)) for val in values]):
            raise ValueError("One or more values invalid for input as hyperparameter.")
        return values
    except:
        return invalid


def mass_evaluate(evaluate, individuals, pset, metadataset: pd.DataFrame, surrogates: typing.Dict[str, object]):
    """ evaluate_fn is only for compatability with `map` signature. """
    fns = [gp.compile(individual, pset) for individual in individuals]
    lengths = [max(n_primitives_in(individual), 0) for individual in individuals]
    scores_full = np.zeros(shape=(len(individuals), len(metadataset)), dtype=float)

    for i, (idx, row) in enumerate(metadataset.iterrows()):
        hyperparam_values = [evaluate(fn, row) for fn in fns]
        surrogate = surrogates[idx]
        scores = surrogate.predict(hyperparam_values)
        scores_full[:, i] = scores

    scores_mean = scores_full.mean(axis=1)  # row wise
    return zip(scores_mean, lengths)


def random_mutation(ind, pset):
    valid_mutations = [
        functools.partial(gp.mutNodeReplacement, pset=pset),
        functools.partial(gp.mutInsert, pset=pset),
        functools.partial(gp.mutEphemeral, mode='all')
    ]

    if n_primitives_in(ind) > 0:
        valid_mutations.append(gp.mutShrink)

    return np.random.choice(valid_mutations)(ind)
