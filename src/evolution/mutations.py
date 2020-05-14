import functools
import random
from inspect import isclass

import numpy as np
from deap import gp

from evolution import Int, Float
from evolution.operations import n_primitives_in


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


def mutTerminalReplacement(individual, pset):
    """Replaces a randomly chosen Terminal from *individual* by a randomly
    chosen Terminal of the same type.

    :param individual: The normal or typed tree to be mutated.
    :returns: A tuple of one tree.
    """
    if len(individual) < 2:
        return individual,

    idxs = [(idx, step) for idx, step in enumerate(individual) if isinstance(step, gp.Terminal)]
    # Sample a Terminal to be replaced
    idx, term = random.choice(idxs)
    same_type = [t for t in pset.terminals[term.ret] if term.ret == t.ret]

    # Replace a Terminal with same type Symbolic
    valid = [t for t in same_type if isinstance(t, gp.Terminal) and t != term]
    if len(valid) > 0 and random.random() > 0.5:
        # Replace a Terminal with same type Symbolic
        individual[idx] = random.choice(valid)
    else:
        # Replace a Terminal with same type Ephemeral
        # Ephemerals contain the class constructor, not thhe class, are therefore of type 'type'
        individual[idx] = [t for t in same_type if isinstance(t, type)][0]()

    return individual,


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


def random_mutation(ind, pset, max_depth=None, toolbox=None):
    valid_mutations = [
        functools.partial(mutNodeReplacement, pset=pset),
        functools.partial(mutTerminalReplacement, pset=pset),
    ]

    if shrinkable_primitives(ind):
        valid_mutations.append(mutShrink)

    if max_depth is None:
        valid_mutations.append(functools.partial(mutInsert, pset=pset))
    elif n_primitives_in(ind) < max_depth:
        valid_mutations.append(functools.partial(mutInsert, pset=pset))

    if get_ephemerals(ind):
        valid_mutations.append(functools.partial(mut_ephemeral_gaussian, pset=pset))

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
