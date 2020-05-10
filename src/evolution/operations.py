import functools
import typing
import random
from abc import ABC
from inspect import isclass

import numpy as np
import pandas as pd
from deap import gp
from deap.gp import mutEphemeral, Primitive
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
        if random.random() < f_kwargs["subset"]:
            # Add metadata and scipy.optimize *args to the dict for evaluation
            metadata = row.to_dict()
            for k,v in enumerate(args):
                metadata.update({f'c_{k}': v})
            hyperparam_values = f_kwargs["evaluate"](fn, metadata)
            scores_full[j] = f_kwargs["surrogates"][idx].predict(np.array(hyperparam_values).reshape(1,-1))

    return - scores_full[scores_full != 0].mean()

def per_individual_evals(evaluate, ind, metadataset: pd.DataFrame, surrogates: typing.Dict[str, object], toolbox, subset=1.0, optimize_constants=False, problem=None):
    fn = numpy_phenotype(ind)
    scores_full = np.zeros(shape=(len(metadataset)), dtype=float)
    # First optimize the mean error across datasets
    opt, sc = const_opt(avg_per_individual_error, ind,
        f_kwargs={"evaluate":evaluate, "metadataset": metadataset, "surrogates":surrogates, "subset":subset},
        method="Nelder-Mead", options={'maxiter':2, 'xatol':1e-4, 'fatol':1e-4})

    # Get results for each dataset
    for j, (idx, row) in enumerate(metadataset.iterrows()):
        metadata = row.to_dict()
        for k,v in enumerate(opt):
            metadata.update({f'c_{k}': v})
        hyperparam_values = evaluate(fn, metadata)
        scores_full[j] = surrogates[idx].predict(np.array(hyperparam_values).reshape(1,-1))

    return scores_full

def mass_evaluate_2(evaluate, individuals, pset, metadataset: pd.DataFrame, surrogates: typing.Dict[str, object], toolbox, subset=1.0, optimize_constants=False, problem=None):
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
        scores_full[i, :] = per_individual_evals(evaluate, ind, metadataset, surrogates, toolbox, subset, optimize_constants, problem)

    scores_mean = scores_full[:, scores_full.sum(axis=0) > 0].mean(axis=1)  # row wise, non-zero columns
    return zip(scores_mean, lengths)

def mass_evaluate(evaluate, individuals, pset, metadataset: pd.DataFrame, surrogates: typing.Dict[str, object], toolbox, subset=1.0, optimize_constants=False, problem=None):
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


def shrinkable_primitives(individual):
    # All primitives are shrinkable, except:
    # - root node
    # - level-one nodes with only float input for int hyperparameter
    # So it is easier to find which primitives can not be shrunk:
    subtrees = _find_depth_one_subtrees(individual)
    shrinkable = {
        i: Float  # we overwrite the expected type if necessary
        for i, node in enumerate(individual[1:], 1)
        if isinstance(node, gp.Primitive)
    }

    for subtree, outtype in zip(subtrees, individual[0].args):
        sub = individual[subtree]
        if isinstance(sub[0], gp.Primitive):
            # Correct the expected output type
            shrinkable[subtree.start] = outtype
            if all([child.ret == Float for child in sub[1:]]) and outtype == Int:
                # Primitive only has Float inputs but requires Int output
                # This primitive can not be shrunk.
                del shrinkable[subtree.start]

    return shrinkable


def random_mutation(ind, pset, max_depth=None, toolbox=None, eph_mutation="gaussian"):
    valid_mutations = [
        functools.partial(mutNodeReplacement, pset=pset),
    ]

    if shrinkable_primitives(ind):
        valid_mutations.append(mutShrink)

    if max_depth is None:
        valid_mutations.append(functools.partial(mutInsert, pset=pset))
    elif n_primitives_in(ind) < max_depth:
        valid_mutations.append(functools.partial(mutInsert, pset=pset))

    if get_ephemerals(ind) and eph_mutation == "gaussian":
        valid_mutations.append(functools.partial(mut_ephemeral_gaussian, pset=pset))

    if get_ephemerals(ind) and eph_mutation == "one":
        valid_mutations.append(functools.partial(mutEphemeral, mode="one"))

    if get_ephemerals(ind) and eph_mutation == "improve":
        valid_mutations.append(functools.partial(mut_small_ephemeral_improve, pset=pset, toolbox=toolbox))

    if get_ephemerals(ind) and eph_mutation == "local":
        valid_mutations.append(functools.partial(mut_small_ephemeral_change, pset=pset))

    if get_ephemerals(ind) and eph_mutation == "many":
        valid_mutations.append(functools.partial(mutEphemeral, mode="one"))
        valid_mutations.append(functools.partial(mutEphemeral, mode="all"))
        valid_mutations.append(functools.partial(mut_ephemeral_gaussian, pset=pset, mode="one"))
        valid_mutations.append(functools.partial(mut_ephemeral_gaussian, pset=pset, mode="all"))
    mut = np.random.choice(valid_mutations)
    # if hasattr(mut, '__name__'):
    #     print(mut.__name__)
    # else:
    #     print(mut.func.__name__)
    return mut(ind)


def get_ephemerals(individual):
    return [(index, node) for index, node in enumerate(individual)
            if isinstance(node, gp.Ephemeral)]


def mut_ephemeral_gaussian(individual, pset, mode="one", s=0.1):
    ephemerals_idx = [(index, node) for index, node in enumerate(individual)
                      if isinstance(node, gp.Ephemeral)]
    if len(ephemerals_idx) > 0:
        if mode == "one":
            index, ephemeral = random.choice(ephemerals_idx)
            ephemerals_idx = [(index, ephemeral)]
        for (index, ephemeral) in ephemerals_idx:
            value = ephemeral.value + random.gauss(0, abs(ephemeral.value) * s)
            if ephemeral.ret == Int:
                # Require a change of at least 1 for integers.
                change = value - ephemeral.value
                if change == 0:
                    change = random.choice([-1, 1])
                if abs(change) < 1:
                    value = ephemeral.value + change/abs(change)
                value = int(value)

            eph = type(ephemeral)()
            eph.value = value
            individual[index] = eph
    return individual,


def mut_small_ephemeral_change(individual, pset):
    ephemerals_idx = [(index, node) for index, node in enumerate(individual)
                      if isinstance(node, gp.Ephemeral)]
    if len(ephemerals_idx) > 0:
        index, ephemeral = random.choice(ephemerals_idx)
        if ephemeral.__class__.__name__ == 'cs':
            # continuous
            value = ephemeral.value + random.gauss(0, 0.1)
        elif ephemeral.value == ephemeral.values[0]:
            value = ephemeral.values[1]
        elif ephemeral.value == ephemeral.values[-1]:
            value = ephemeral.values[-2]
        else:
            value_index = ephemeral.values.index(ephemeral.value)
            lower, higher = ephemeral.values[value_index - 1], ephemeral.values[value_index + 1]
            value = random.choice([lower, higher])
        # Assign the value in a new ephemeral object, so as not to change the old one.
        eph = type(ephemeral)()
        eph.value = value
        individual[index] = eph
    return individual,


# If it is not at edge, look up ephemeral next and compare both
# If it is at edge, look up difeferent eph types that are close and look up their neighbors.

def mut_small_ephemeral_improve(individual, pset, toolbox):
    ephemerals_idx = [(index, node) for index, node in enumerate(individual)
                      if isinstance(node, gp.Ephemeral)]
    if len(ephemerals_idx) == 0:
        return

    index, ephemeral = random.choice(ephemerals_idx)
    if ephemeral.__class__.__name__ == 'cs':
        v1 = max(ephemeral.value - 0.1, 0)
        v2 = min(ephemeral.value + 0.1, 1.0)
    elif ephemeral.value == ephemeral.values[0]:
        v1 = ephemeral.values[1]
        v2 = ephemeral.values[2]
    elif ephemeral.value == ephemeral.values[-1]:
        v1 = ephemeral.values[-2]
        v2 = ephemeral.values[-3]
    else:
        value_index = ephemeral.values.index(ephemeral.value)
        v1, v2 = ephemeral.values[value_index - 1], ephemeral.values[value_index + 1]

    # Create two new individual objects with the distinct ephemerals
    ind_1, ind_2 = toolbox.clone(individual), toolbox.clone(individual)
    eph1, eph2 = type(ephemeral)(), type(ephemeral)()
    eph1.value, eph2.value = v1, v2
    ind_1[index], ind_2[index] = eph1, eph2

    fitnesses = toolbox.map(toolbox.evaluate, [ind_1, ind_2])
    best = np.argmax(fitnesses)
    return [ind_1, ind_2][best],


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


def approx_eq(a, b, eps = 1e-3):
    a, asz = a.fitness.values
    b, bsz = b.fitness.values
    if abs(a - b) / abs(b) < eps and asz == bsz:
        return True
    else:
        return False


def _find_depth_one_subtrees(ind):
    slices = [ind.searchSubtree(1)]
    while slices[-1].stop != len(ind):
        slices.append(ind.searchSubtree(slices[-1].stop))
    return slices


def cxDepthOne(ind1, ind2, n=None):
    """ Randomly select a depth-one subtree to swap with the same subtree of the other.

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :param n: the number of subtrees to crossover. If None, select a random number.
    :returns: A tuple of two trees.
    """
    slices1 = _find_depth_one_subtrees(ind1)
    slices2 = _find_depth_one_subtrees(ind2)

    if n is None:
        n = random.randint(1, len(slices1) - 1)

    cx_subtrees = random.sample(range(len(slices1)), n)
    for i in reversed(sorted(cx_subtrees)):
        ind1[slices1[i]], ind2[slices2[i]] = ind2[slices2[i]], ind1[slices1[i]]
    return ind1, ind2


def mutNodeReplacement(individual, pset):
    """Replaces a randomly chosen primitive from *individual* by a randomly
    chosen primitive with the same number of arguments from the :attr:`pset`
    attribute of the individual.

    :param individual: The normal or typed tree to be mutated.
    :returns: A tuple of one tree.
    """
    if len(individual) < 2:
        return individual,

    while True:
        index = random.randrange(1, len(individual))
        node = individual[index]
        if node.arity == 0:  # Terminal
            # Don't replace a terminal by the same terminal, except for Ephemerals, since
            # for those a new value is sampled anyway.
            terms = [t for t in pset.terminals[node.ret]
                     if t != node or isinstance(t, gp.Ephemeral)]
            # There is always a valid option to pick from
            term = random.choice(terms)
            if isclass(term):
                term = term()
            individual[index] = term
            break
        else:  # Primitive
            # This replacement works out of the box, as all primitives (except make_tuple)
            # have the same signature (ignoring input arity).
            prims = [
                p
                for p in pset.primitives[node.ret]
                if p != node and p.args == node.args
            ]
            if prims:
                individual[index] = random.choice(prims)
                break
            # Some primitives don't have alternatives, just try mutation again

    return individual,


def mutInsert(individual, pset):
    """Inserts a new branch at a random position in *individual*. The subtree
    at the chosen position is used as child node of the created subtree, in
    that way, it is really an insertion rather than a replacement. Note that
    the original subtree will become one of the children of the new primitive
    inserted, but not perforce the first (its position is randomly selected if
    the new primitive has more than one child).

    :param individual: The normal or typed tree to be mutated.
    :returns: A tuple of one tree.
    """
    indices = [
        i for i, node in enumerate(individual)
        if len(pset.primitives[node.ret]) > 1
    ]
    index = random.choice(indices)
    node = individual[index]
    slice_ = individual.searchSubtree(index)
    choice = random.choice

    # As we want to keep the current node as children of the new one,
    # it must accept the return value of the current node
    # If the primitive only as Float input, this should still be valid as
    # Int is a subclass of Float.
    primitives = [
        p
        for p in pset.primitives[node.ret]
        if any([issubclass(node.ret, a) for a in p.args])
    ]

    if len(primitives) == 0:
        return individual,

    new_node = choice(primitives)
    new_subtree = [None] * len(new_node.args)
    # again allow for subclass
    position = choice([i for i, a in enumerate(new_node.args) if issubclass(node.ret, a)])

    for i, arg_type in enumerate(new_node.args):
        if i != position:
            term = choice(pset.terminals[arg_type])
            if isclass(term):
                term = term()
            new_subtree[i] = term

    new_subtree[position:position + 1] = individual[slice_]
    new_subtree.insert(0, new_node)
    individual[slice_] = new_subtree
    return individual,


def mutShrink(individual):
    """This operator shrinks the *individual* by choosing randomly a branch and
    replacing it with one of the branch's arguments (also randomly chosen).

    :param individual: The tree to be shrinked.
    :returns: A tuple of one tree.
    """
    # We don't want to "shrink" the root
    if len(individual) < 3 or individual.height <= 1:
        return individual,

    candidates = shrinkable_primitives(individual)
    iprims = []
    for i, outtype in candidates.items():
        iprims.append((i, individual[i], outtype))

    if len(iprims) != 0:
        index, prim, outtype = random.choice(iprims)
        prim_subtree = individual.searchSubtree(index)
        # print(str(individual), index, str(prim))
        # Find subtrees of this primitive
        slices = [individual.searchSubtree(index+1)]
        while slices[-1].stop != prim_subtree.stop:
            slices.append(individual.searchSubtree(slices[-1].stop))
        # Filter to only subtrees that provide Int if needed:
        if outtype == Int:
            slices = [s for s in slices if individual[s.start].ret == Int]

        replacement = random.choice(slices)
        subtree = individual[replacement]
        individual[prim_subtree] = subtree

    return individual,
