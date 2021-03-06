import functools
import operator
import random
import typing

import numpy as np
import scipy.special

from deap import gp, base, creator, tools

from .operations import try_evaluate_function, Int, Float
from evolution.mutations import cxDepthOne, random_mutation


def if_gt(a, b, x, y):
    return x if a > b else y


def poly_gt(a, b):
    # Introduced specifically for polynomial kernel degree hyperparameter.
    return 1. if a > b else 3.


def setup_toolbox(problem, args):

    if not args.constants_only:
        # Set variables of our genetic program:
        variables = dict(
            NumberOfClasses=('m', Int),           #  [2;50]
            NumberOfFeatures=('p', Int),          #  [1;Inf]
            NumberOfFeaturesBefore=('po', Int),   #  [1:Inf]
            NumberOfInstances=('n', Int),         #  [1;Inf]
            MedianKernelDistance=('mkd', Float),    #  [0;Inf]
            MajorityClassPercentage=('mcp', Float), #  [0;1]
            RatioSymbolicFeatures=('rc', Float),    #  [0;1]   'ratio categorical' := #Symbolic / #Features
            Variance=('xvar', Float),                #  [0;Inf] variance of all elements
        )

        variable_names, variable_types = zip(*list(variables.values()))
        pset = gp.PrimitiveSetTyped("SymbolicExpression", variable_types, typing.Tuple)
        pset.renameArguments(**{f"ARG{i}": var for i, var in enumerate(variable_names)})
    else:
        pset = gp.PrimitiveSetTyped("ConstantExpression", [], typing.Tuple)

    if args.optimize_constants:
        pset.args = pset.arguments
        symc = 1.0
        pset.addTerminal(symc, Float, "Symc")
        pset.constants = ["Symc"]
    else:
        def loguniform_float():
            return 2 ** random.uniform(-10, 0)
        pset.addEphemeralConstant("F", loguniform_float, ret_type=Float)
        def loguniform_int():
            return round(2 ** random.uniform(0, 10))
        pset.addEphemeralConstant("I", loguniform_int, ret_type=Int)

    pset.addPrimitive(if_gt, [Float, Float, Float, Float], Int)
    # pset.addPrimitive(poly_gt, [float, float], float)
    binary_operators = [operator.add, operator.mul, operator.sub, operator.truediv, operator.pow, max, min]
    unary_operators = [scipy.special.expit, operator.neg]
    for binary_operator in binary_operators:
        pset.addPrimitive(binary_operator, [Float, Float], Int)
    for unary_operator in unary_operators:
        pset.addPrimitive(unary_operator, [Float], Int)

    # Finally add a 'root-node' primitive (needed to find three functions at once)
    def make_tuple(*args):
        return tuple([*args])

    n_hyperparams = len(problem.hyperparameters) - len(problem.fixed)
    hp_types = [
        Int if t == "int" else Float
        for hp, t in problem.hyperparameter_types.items()
        if hp not in problem.fixed
    ]
    pset.addPrimitive(make_tuple, hp_types, typing.Tuple)

    # More DEAP boilerplate...
    creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, birthyear=None, __hash__=lambda x: hash(str(x)), __eq__=lambda s, o: hash(s) == hash(o))

    def initBenchmarkPopulation(pcls, ind_init, pset, problem):
        return pcls(ind_init(gp.PrimitiveTree.from_string(c, pset)) for c in problem.benchmarks.values())

    def initSymcPopulation(pcls, ind_init, pset, problem):
        c = f'make_tuple({",".join(["Symc"]*n_hyperparams)})'
        return pcls([ind_init(gp.PrimitiveTree.from_string(c, pset))])

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=args.max_start_size+1)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population_symc", initSymcPopulation, list, creator.Individual, pset)
    toolbox.register("population_benchmark", initBenchmarkPopulation, list, creator.Individual, pset)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", cxDepthOne)
    toolbox.register("mutate", random_mutation, pset=pset, max_depth=args.max_number_operators, toolbox=toolbox)

    # We abuse 'map/evaluate'.
    # Here 'evaluate' computes the hyperparameter values based on the symbolic
    # expression and meta-feature values. Then 'map' will evaluate all these
    # configurations in one batch with the use of surrogate models.
    toolbox.register("evaluate", functools.partial(try_evaluate_function,
                                                   invalid=(1e-6,) * n_hyperparams,
                                                   problem=problem))

    toolbox.decorate("mate",   gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))

    return toolbox, pset
