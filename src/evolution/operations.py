import typing
import random
from abc import ABC

import numpy as np
import pandas as pd
from deap import gp
from glyph.gp import numpy_phenotype
from glyph.assessment import const_opt
from glyph.utils import Memoize


class Float(ABC):
    @classmethod
    def __subclasshook__(cls, C):
        return C in [cls, Int, float]


class Int(Float):
    pass


def n_primitives_in(individual):
    """ Return the number of primitives in the individual. """
    return len([e for e in individual if isinstance(e, gp.Primitive)])


def insert_fixed(hyperparam_values, problem):
    """
    insert problem.fixed (a dict of fixed hyperparameter values, e.g. nrounds: 500) into the
    hyperparameter values according to its position in problem.hyperparameters.
    """
    if len(hyperparam_values) == len(problem.hyperparameters):
        return hyperparam_values
    else:
        hp = hyperparam_values
        for key, val in problem.fixed.items():
            n = problem.hyperparameters.index(key)
            hp = hp[:n] + (val,) + hp[n:]
        return hp


def try_compile_individual(ind, pset, problem, invalid=1e-6):
    """
       For constant optimization gp.compile already returns a tuple
       This tuple can be invalid / raise erros, therefore we have to catch here.
    """
    try:
        fn = gp.compile(ind, pset)
        if callable(fn):
            return fn
        else:
            if not all([not isinstance(val, complex) and abs(val) < 1.2676506e+30 for val in fn]):
                raise ValueError("One or more values invalid for input as hyperparameter.")
        return insert_fixed(fn, problem)
    except:
        return insert_fixed((invalid, )*(len(problem.hyperparameters)-len(problem.fixed)), problem)


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
    # If individual has only constants, fn is not afunction but already has the results.
    if not callable(fn):
        return(fn)

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
    """
    Compute the fitness of a individual as the average over datasets
    :param *args:
         Numeric args passed on from scikit.optimize.minimize
         (This is what optimize_constants optimizes)
    :param **f_kwarrgs :
        Further arguments required for evaluation
    """
    fn = numpy_phenotype(ind)
    metadataset = f_kwargs["metadataset"]
    scores_full = np.zeros(shape=(len(metadataset)), dtype=float)
    for j, (idx, row) in enumerate(metadataset.iterrows()):
        # Add metadata and scipy.optimize *args to the dict for evaluation
        metadata = row.to_dict()
        for k,v in enumerate(args):
            metadata.update({f'c_{k}': v})
        hyperparam_values = f_kwargs["evaluate"](fn, metadata)
        scores_full[j] = f_kwargs["surrogates"][idx].predict(np.array(hyperparam_values).reshape(1,-1))

    return - scores_full[scores_full != 0].mean()


def per_individual_evals(evaluate, ind, metadataset: pd.DataFrame, surrogates: typing.Dict[str, object], toolbox, optimize_constants=False, problem=None):
    fn = numpy_phenotype(ind)
    scores_full = np.zeros(shape=(len(metadataset)), dtype=float)
    # First optimize the mean error across datasets
    opt, sc = const_opt(avg_per_individual_error, ind,
        f_kwargs={"evaluate":evaluate, "metadataset": metadataset, "surrogates":surrogates},
        method="Nelder-Mead", options={'maxiter':2, 'xatol':1e-4, 'fatol':1e-4})

    # Get results for each dataset
    for j, (idx, row) in enumerate(metadataset.iterrows()):
        metadata = row.to_dict()
        for k,v in enumerate(opt):
            metadata.update({f'c_{k}': v})
        hyperparam_values = evaluate(fn, metadata)
        scores_full[j] = surrogates[idx].predict(np.array(hyperparam_values).reshape(1,-1))

    return scores_full


def mass_evaluate_2(evaluate, individuals, pset, metadataset: pd.DataFrame, surrogates: typing.Dict[str, object], toolbox, optimize_constants=False, problem=None):
    """
    Evaluate all individuals by averaging their projected score on each dataset using surrogate models.
    :param evaluate: should turn (fn, row) into valid hyperparameter values.
    """
    if individuals == []:
        return []

    lengths = [max(n_primitives_in(individual), 0) for individual in individuals]
    scores_full = np.zeros(shape=(len(individuals), len(metadataset)), dtype=float)

    for i, ind in enumerate(individuals):
        ind.terminals = [p for p in ind if p.arity == 0] # hackery to make glyph happy
        ind.pset = pset # hackery to make glyph happy
        scores_full[i, :] = per_individual_evals(evaluate, ind, metadataset, surrogates, toolbox, optimize_constants, problem)

    scores_mean = scores_full[:, scores_full.sum(axis=0) > 0].mean(axis=1)  # row wise, non-zero columns
    return zip(scores_mean, lengths)


def mass_evaluate(evaluate, individuals, pset, metadataset: pd.DataFrame, surrogates: typing.Dict[str, object], toolbox, optimize_constants=False, problem=None):
    """
        Evaluate all individuals by averaging their projected score on each dataset using surrogate models.
        :param evaluate: should turn (fn, row) into valid hyperparameter values.
    """
    if individuals == []:
        return []

    fns = []
    for ind in individuals:
        fns.append(try_compile_individual(ind, pset, problem))

    lengths = [max(n_primitives_in(individual), 0) for individual in individuals]
    scores_full = np.zeros(shape=(len(individuals), len(metadataset)), dtype=float)

    for i, (idx, row) in enumerate(metadataset.iterrows()):
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


def approx_eq(a, b, eps=1e-3):
    a, asz = a.fitness.values
    b, bsz = b.fitness.values
    if abs(a - b) / abs(b) < eps and asz == bsz:
        return True
    else:
        return False
