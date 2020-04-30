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

    if not args.constants_only:
        # Set variables of our genetic program:
        variables = dict(
            NumberOfClasses='m',           #  [2;50]
            NumberOfFeatures='p',          #  [1;Inf]
            NumberOfFeaturesBefore='po',
            NumberOfInstances='n',         #  [1;Inf]
            MedianKernelDistance='mkd',    #  [0;Inf]
            MajorityClassPercentage='mcp', #  [0;1]
            RatioSymbolicFeatures='rc',    #  [0;1]   'ratio categorical' := #Symbolic / #Features
            Variance='xvar'                #  [0;Inf] variance of all elements
        )
        variable_names = list(variables.values())
        pset = gp.PrimitiveSetTyped("SymbolicExpression", [float] * len(variables), typing.Tuple)
        pset.renameArguments(**{f"ARG{i}": var for i, var in enumerate(variable_names)})
    else:
        pset = gp.PrimitiveSetTyped("ConstantExpression", [], typing.Tuple)

    if args.optimize_constants:
        pset.args = pset.arguments
        symc = 1.0
        pset.addTerminal(symc, float, "Symc")
        pset.constants = ["Symc"]
    else:
        cloggt1 = [2 ** i for i in range(4, 11)]+[10 ** i for i in range(1, 4)]
        cloglt1 = [2 ** i for i in range(-8, -1)]+[10 ** i for i in range(-4, -1)]
        pset.addEphemeralConstant("cs", lambda: random.random(), ret_type=float)
        pset.addEphemeralConstant("ci", lambda: float(random.randint(1, 10)), ret_type=float)
        pset.addEphemeralConstant("cloggt1", lambda: np.random.choice(cloggt1), ret_type=float)
        pset.addEphemeralConstant("cloglt1", lambda: np.random.choice(cloglt1), ret_type=float)

        # Ephemerals don't have data about their ranges out of the box.
        # We add these so that we can later perform small changes
        ranges = dict(
            cs=[i / 10 for i in range(1, 11)],
            ci=[i for i in range(1, 11)],
            cloggt1=list(sorted(cloggt1)),
            cloglt1=list(sorted(cloglt1)),
        )
        ephemerals = [t for t in pset.terminals[float] if hasattr(t, '__name__')]
        for e in ephemerals:
            e.values = ranges[e.__name__]

    pset.addPrimitive(if_gt, [float, float, float, float], float)
    # pset.addPrimitive(poly_gt, [float, float], float)
    binary_operators = [operator.add, operator.mul, operator.sub, operator.truediv, operator.pow, max, min]
    unary_operators = [scipy.special.expit, operator.neg]
    for binary_operator in binary_operators:
        pset.addPrimitive(binary_operator, [float, float], float)
    for unary_operator in unary_operators:
        pset.addPrimitive(unary_operator, [float], float)

    # Finally add a 'root-node' primitive (needed to find three functions at once)
    def make_tuple(*args):
        return tuple([*args])

    n_hyperparams = len(problem.hyperparameters) - len(problem.fixed)
    pset.addPrimitive(make_tuple, [float] * n_hyperparams, typing.Tuple)

    # More DEAP boilerplate...
    creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, birthyear=0)

    def initBenchmarkPopulation(pcls, ind_init, pset, problem):
        return pcls(ind_init(gp.PrimitiveTree.from_string(c, pset)) for c in problem.benchmarks.values())

    def initSymcPopulation(pcls, ind_init, pset, problem):
        c = f'make_tuple({",".join(["Symc"]*n_hyperparams)})'
        return pcls([ind_init(gp.PrimitiveTree.from_string(c, pset))])

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population_symc", initSymcPopulation, list, creator.Individual, pset)
    toolbox.register("population_benchmark", initBenchmarkPopulation, list, creator.Individual, pset)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", random_mutation, pset=pset, max_depth=args.max_number_operators, toolbox=toolbox, eph_mutation=args.emut)

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
