import functools
import typing
import random

import numpy as np
import pandas as pd
from deap import gp
from glyph.gp import numpy_phenotype
from glyph.assessment import const_opt
from glyph.utils import Memoize

def n_primitives_in(individual):
    """ Return the number of primitives in the individual. """
    return len([e for e in individual if isinstance(e, gp.Primitive)])

def insert_fixed(hyperparam_values, problem):
    """
    insert problem.fixed (a dict of fixed hyperparameter values, e.g. nrounds: 500) into the 
    hyperparameter values according to its position in problem.hyperparameters.
    """ 
    hp = hyperparam_values
    for key, val in problem.fixed.items():
        n = problem.hyperparameters.index(key)
        hp = hp[:n] + (val,) + hp[n:]
    return hp

def try_evaluate_function(fn, input_, invalid, problem=None):
    """ Return fn(input_) if output is sequence of finite float32, else return `invalid`

    This function is used to evaluate the symbolic expressions for a particular dataset.

    :param fn: callable.
        Takes input `input_` and should return a sequence of numbers.
    :param input_:
        input arguments to `fn`
    :param invalid:
        return value if `fn(input_)` does not return a sequence of finite float32 values
    :return:
        fn(input_) if its output is a sequence of finite float32 values, else `invalid`
    """
    try:
        values = fn(**dict(input_))
        # All float values need to be finite in range float32,
        # i.e. abs() < 3.4028235e+38 (np.finfo(np.float32).max)
        # However, `predict` internally also sums all values,
        # and this may incur an overflow even if each individual value is finite.
        # We constrict ourselves to the much smaller value of 2**100
        # so that the summation should also never lead to an overflow value.
        if not all([not isinstance(val, complex) and abs(val) < 1.2676506e+30
                    for val in values]):
            raise ValueError("One or more values invalid for input as hyperparameter.")
        return insert_fixed(values, problem)
    except:
        return insert_fixed(invalid, problem)

@Memoize
def avg_per_individual_error(ind, *args, **f_kwargs):  
    fn = numpy_phenotype(ind)
    metadataset = f_kwargs["metadataset"]
    scores_full = np.zeros(shape=(len(metadataset)), dtype=float)
    for j, (idx, row) in enumerate(metadataset.iterrows()):
        if random.random() < f_kwargs["subset"]:
            metadata = row.to_dict()
            for k,v in enumerate(args):
                metadata.update({f'c_{k}': v})    
            hyperparam_values = f_kwargs["evaluate"](fn, metadata)
            scores_full[j] = f_kwargs["surrogates"][idx].predict(np.array(hyperparam_values).reshape(1,-1))

    return - scores_full[scores_full != 0].mean()

def per_individual_evals(evaluate, ind, metadataset: pd.DataFrame, surrogates: typing.Dict[str, object], toolbox, subset=1.0, optimize_constants=False, problem=None):
    fn = numpy_phenotype(ind)
    scores_full = np.zeros(shape=(len(metadataset)), dtype=float)
    opt, sc = const_opt(avg_per_individual_error, ind, f_kwargs={"evaluate":evaluate, "metadataset": metadataset, "surrogates":surrogates, "subset":subset}, method="Nelder-Mead")

    for j, (idx, row) in enumerate(metadataset.iterrows()):
        metadata = row.to_dict()
        for k,v in enumerate(opt):
            metadata.update({f'c_{k}': v})    
        hyperparam_values = evaluate(fn, metadata)
        scores_full[j] = surrogates[idx].predict(np.array(hyperparam_values).reshape(1,-1))

    return scores_full

def mass_evaluate_2(evaluate, individuals, pset, metadataset: pd.DataFrame, surrogates: typing.Dict[str, object], toolbox, subset=1.0, optimize_constants=False, problem=None):
    """ Evaluate all individuals by averaging their projected score on each dataset using surrogate models.
    :param evaluate: should turn (fn, row) into valid hyperparameter values. """
    if individuals == []:
        return []

    lengths = [max(n_primitives_in(individual), 0) for individual in individuals]
    scores_full = np.zeros(shape=(len(individuals), len(metadataset)), dtype=float)

    for i, ind in enumerate(individuals):
        ind.terminals = [p for p in ind if p.arity == 0] # hackery to make glyph happy
        ind.pset = pset # hackery to make glyph happy
        scores_full[i, :] = per_individual_evals(evaluate, ind, metadataset, surrogates, toolbox, subset, optimize_constants, problem)

    scores_mean = scores_full[:, scores_full.sum(axis=0) > 0].mean(axis=1)  # row wise, non-zero columns
    return zip(scores_mean, lengths)

def mass_evaluate(evaluate, individuals, pset, metadataset: pd.DataFrame, surrogates: typing.Dict[str, object], toolbox, subset=1.0, optimize_constants=False, problem=None):
    """ Evaluate all individuals by averaging their projected score on each dataset using surrogate models.
    :param evaluate: should turn (fn, row) into valid hyperparameter values. """
    if individuals == []:
        return []

    fns = []
    for ind in individuals:
        if optimize_constants:
            # For now ignore the fact that we also duplicate individuals with no constants.
            # It should not affect results even though it will increase compute cost.
            variations = [mut_all_constants(toolbox.clone(ind), pset) for _ in range(5)]
            fns += [gp.compile(variation, pset) for variation in variations]
        else:
            fns.append(gp.compile(ind, pset))

    lengths = [max(n_primitives_in(individual), 0) for individual in individuals]
    scores_full = np.zeros(shape=(len(individuals), len(metadataset)), dtype=float)

    for i, (idx, row) in enumerate(metadataset.iterrows()):
        if random.random() < subset:
            hyperparam_values = [evaluate(fn, row) for fn in fns]
            surrogate = surrogates[idx]
            scores = surrogate.predict(hyperparam_values)
            evaluations_per_individual = len(scores) / len(individuals)
            if evaluations_per_individual > 1:
                scores = [max(scores[int(i * evaluations_per_individual):
                                     int(i * evaluations_per_individual + evaluations_per_individual)])
                          for i in range(len(individuals))]
            scores_full[:, i] = scores

    scores_mean = scores_full[:, scores_full.sum(axis=0) > 0].mean(axis=1)  # row wise, non-zero columns
    return zip(scores_mean, lengths)


def random_mutation(ind, pset, max_depth=None):
    valid_mutations = [
        functools.partial(gp.mutNodeReplacement, pset=pset),
        functools.partial(gp.mutEphemeral, mode='all')
    ]

    if n_primitives_in(ind) > 1:  # The base primitive is unshrinkable.
        valid_mutations.append(gp.mutShrink)

    if max_depth is None:
        valid_mutations.append(functools.partial(gp.mutInsert, pset=pset))
    elif n_primitives_in(ind) < max_depth:
        valid_mutations.append(functools.partial(gp.mutInsert, pset=pset))

    return np.random.choice(valid_mutations)(ind)


def mut_all_constants(individual, pset):
    # Ephemerals have a non-string name...
    ephemerals = [el for el in pset.terminals[float] if not isinstance(el.name, str)]

    ephemerals_idx = [index
                      for index, node in enumerate(individual)
                      if isinstance(node, gp.Ephemeral)]

    if len(ephemerals_idx) > 0:
        for i in ephemerals_idx:
            individual[i] = random.choice(ephemerals)()

    return individual
