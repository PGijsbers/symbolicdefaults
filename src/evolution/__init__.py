import functools
import operator
import random
import typing

import numpy as np
import scipy.special

from deap import gp, base, creator, tools

from .operations import random_mutation, try_evaluate_function


def if_gt(a, b, x, y):
    return x if a > b else y


def poly_gt(a, b):
    # Introduced specifically for polynomial kernel degree hyperparameter.
    return 1. if a > b else 3.


def setup_toolbox(problem, args):
    # Set variables of our genetic program:
    variables = dict(
        NumberOfClasses='m',           #  [2;50]
        NumberOfFeatures='p',          #  [1;Inf]
        NumberOfInstances='n',         #  [1;Inf]
        MedianKernelDistance='mkd',    #  [0;Inf]
        MajorityClassPercentage='mcp', #  [0;1]
        RatioSymbolicFeatures='rc',    #  [0;1]   'ratio categorical' := #Symbolic / #Features
        Variance='xvar'                #  [0;Inf] variance of all elements
    )

    variable_names = list(variables.values())

    pset = gp.PrimitiveSetTyped("SymbolicExpression", [float] * len(variables), typing.Tuple)
    pset.renameArguments(**{f"ARG{i}": var for i, var in enumerate(variable_names)})
    pset.addEphemeralConstant("cs", lambda: random.random(), ret_type=float)
    pset.addEphemeralConstant("ci", lambda: float(random.randint(1, 10)), ret_type=float)
    pset.addEphemeralConstant("clog", lambda: np.random.choice([2 ** i for i in range(-8, 11)]), ret_type=float)

    pset.addPrimitive(if_gt, [float, float, float, float], float)
    pset.addPrimitive(poly_gt, [float, float], float)
    binary_operators = [operator.add, operator.mul, operator.sub, operator.truediv, operator.pow, max, min]
    unary_operators = [scipy.special.expit, operator.neg]
    for binary_operator in binary_operators:
        pset.addPrimitive(binary_operator, [float, float], float)
    for unary_operator in unary_operators:
        pset.addPrimitive(unary_operator, [float], float)

    # Finally add a 'root-node' primitive (needed to find three functions at once)
    def make_tuple(*args):
        return tuple([*args])

    n_hyperparams = len(problem.hyperparameters)
    pset.addPrimitive(make_tuple, [float] * n_hyperparams, typing.Tuple)

    # More DEAP boilerplate...
    creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", random_mutation, pset=pset, max_depth=args.max_number_operators)

    # We abuse 'map/evaluate'.
    # Here 'evaluate' computes the hyperparameter values based on the symbolic
    # expression and meta-feature values. Then 'map' will evaluate all these
    # configurations in one batch with the use of surrogate models.
    toolbox.register("evaluate", functools.partial(try_evaluate_function,
                                                   invalid=(1e-6,) * n_hyperparams))

    toolbox.decorate("mate",   gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))

    return toolbox, pset