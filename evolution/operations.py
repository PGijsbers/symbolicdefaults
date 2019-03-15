import functools
import typing

import numpy as np
import pandas as pd
from deap import gp


def n_primitives_in(individual):
    return len([e for e in individual if isinstance(e, gp.Primitive)])


def mass_evaluate(evaluate_fn, individuals, pset, metadataset: pd.DataFrame, surrogates: typing.Dict[str, object]):
    """ evaluate_fn is only for compatability with `map` signature. """
    fns = [gp.compile(individual, pset) for individual in individuals]
    lengths = [max(n_primitives_in(individual), 0) for individual in individuals]
    scores_full = np.zeros(shape=(len(individuals), len(metadataset)), dtype=float)

    for i, (idx, row) in enumerate(metadataset.iterrows()):
        def try_else_invalid(fn, input_):
            try:
                values = fn(**dict(row))
                if not all([np.isfinite(np.float32(val)) for val in values]):
                   raise ValueError("One or more values invalid for input as hyperparameter.")
                return values
            except:
                return (-1e6, -1e6, -1e6)

        hyperparam_values = [try_else_invalid(fn, row) for fn in fns]

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
