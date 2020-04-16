# Symbolic Defaults

Script to find symbolic expressions for default hyperparameter values.

usage
-
Example: `python main.py adaboost -ngen 10`

```
usage: main.py [-h] [-m MU] [-l LAMBDA_] [-ngen NGEN] [-c CONFIG_FILE] [-esn EARLY_STOPPING_N] problem

Uses evolutionary optimization to find symbolic expressions for default hyperparameter values.

positional arguments:
  problem               Problem to optimize. Must match one of 'name' fields in the configuration file.

optional arguments:
  -h, --help            show this help message and exit
  -m MU                 mu for the mu+lambda algorithm. Specifies the number of individuals that can create offspring.
  -l LAMBDA_            lambda for the mu+lambda algorithm. Specifies the number of offspring created at each iteration.
                        Also used to determine the size of starting population.
  -ngen NGEN            Number of generations.
  -c CONFIG_FILE        Configuration file.
  -esn EARLY_STOPPING_N Early Stopping N. Stop optimization if there is no improvement in n generations.
 ```


# Setup:

## Symbolic Regression

#### Terminals

Metafeatures:
```
NumberOfClasses='m',           #  [2;50]
NumberOfFeatures='p',          #  [1;Inf]
NumberOfInstances='n',         #  [1;Inf]
MedianKernelDistance='mkd',    #  [0;Inf]
MajorityClassPercentage='mcp', #  [0;1]
RatioSymbolicFeatures='rc',    #  [0;1]   'ratio categorical' := #Symbolic / #Features
Variance='xvar'                #  [0;Inf] variance of all elements
```

Ephemeral Constants:
```
"cs":     random.random() # [0,1]
"ci":     random.randint(1, 10) # {1,2, ..., 10}
# {16,32,64, ...,1024, 10, 1e2, 1e3, 1e4}
"cloggt1": np.random.choice([2 ** i for i in range(4, 11)]+[10 ** i for i in range(1, 4)])
# {.5, .25, .125, ..., 0.002, 1e-1, 1e-2, 1e-3, 1e-4}
"cloglt1": np.random.choice([2 ** i for i in range(-9, -1)]+[10 ** i for i in range(-4, -1)])
```

For constant optimization: **Symc**: *c_1, ... c_n* optimized via `scipy.optimize.minimize` via Nelder-Mead

#### Functions

* binary_operators = [operator.add, operator.mul, operator.sub, operator.truediv, operator.pow, max, min]

* unary_operators = [scipy.special.expit, operator.neg]

* if_gt([float, float, float, float]): float
  if (a > b) return c else return d

