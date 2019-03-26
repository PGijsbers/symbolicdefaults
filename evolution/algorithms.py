
def one_plus_lambda(toolbox, P=None, popsize=50, new_per_gen=50, ngen=100, halloffame=None, stats=None, logbook=None):
    P = P if P is not None else toolbox.individual()

    for i in range(ngen):
        # evaluate population
        C = [toolbox.mutate(toolbox.clone(P))[0] for _ in range(popsize)]
        C = C + [toolbox.individual() for _ in range(new_per_gen)]
        C = C + [P]

        fitnesses = toolbox.map(toolbox.evaluate, C)
        for ind, fit in zip(C, fitnesses):
            ind.fitness.values = fit

        P = max(C, key=lambda ind: ind.fitness.wvalues)

        if halloffame is not None:
            halloffame.update(C)

        if logbook is not None:
            record = stats.compile(C) if stats is not None else {}
            logbook.record(gen=i, nevals=len(C), **record)

    return P, C
