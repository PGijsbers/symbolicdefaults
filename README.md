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