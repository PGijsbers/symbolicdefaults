# Symbolic Defaults

Script to find symbolic expressions for default hyperparameter values.

usage
-
```
usage: python main.py [-h] [-m MU] [-l LAMBDA] [-ngen NGEN] [-c CONFIG_FILE] problem

Uses evolutionary optimization to find symbolic expressions for default
hyperparameter values.

positional arguments:
  problem         Problem to optimize. Must match one of 'name' fields in the configuration file.

optional arguments:
  -h, --help      show this help message and exit
  -m MU           mu for the mu+lambda algorithm. Specifies the number of
                  individuals that can create offspring.
  -l LAMBDA       lambda for the mu+lambda algorithm. Specifies the number of
                  offspring created at each iteration.Also used to determine
                  the size of starting population.
  -ngen NGEN      Number of generations.
  -c CONFIG_FILE  Configuration file.
 ```