import functools
import operator
import random
import typing

import numpy as np
import scipy.special

from deap import gp, base, creator, tools

from .operations import random_mutation, try_evaluate_function


def setup_toolbox(problem):
    # Set variables of our genetic program:
    variables = dict(
        NumberOfClasses='m',
        NumberOfFeatures='p',
        NumberOfInstances='n',
        MedianKernelDistance='mkd',
        MajorityClassPercentage='mcp',
        RatioSymbolicFeatures='rc',  # Number Symbolic / Number Features
        Variance='xvar'  # variance of all elements
    )

    variable_names = list(variables.values())

    pset = gp.PrimitiveSetTyped("SymbolicExpression", [float] * len(variables), typing.Tuple)
    pset.renameArguments(**{"ARG{}".format(i): var for i, var in enumerate(variable_names)})
    pset.addEphemeralConstant("cs", lambda: random.random(), ret_type=float)
    pset.addEphemeralConstant("ci", lambda: float(random.randint(1, 10)), ret_type=float)
    pset.addEphemeralConstant("clog", lambda: np.random.choice([2 ** i for i in range(-8, 9)]), ret_type=float)

    binary_operators = [operator.add, operator.mul, operator.sub, operator.truediv, operator.pow, max, min]
    unary_operators = [scipy.special.expit, operator.neg]
    for binary_operator in binary_operators:
        pset.addPrimitive(binary_operator, [float, float], float)
    for unary_operator in unary_operators:
        pset.addPrimitive(unary_operator, [float], float)

    # Finally add a 'root-node' primitive (needed to find three functions at once)
    def make_tuple(*args):
        return tuple([*args])

    pset.addPrimitive(make_tuple, [float] * len(problem['hyperparameters']), typing.Tuple)

    # More DEAP boilerplate...
    creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", random_mutation, pset=pset)

    toolbox.register("evaluate", functools.partial(try_evaluate_function,
                                                   invalid=(1e-6,) * len(problem['hyperparameters'])))

    return toolbox, pset
