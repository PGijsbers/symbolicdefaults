## Problem Definitions

A problem is defined by the hyperparameters to find symbolic defaults for, and the
experiment data to create surrogate models with.
The following fields can be set (optional fields indicated with an asterisk (*)):

* name: Recognizable name for the problem, used from the command line.
* experiment_data: File with experiment data of the algorithm on various tasks.
* metadata: File with metadata for each task on which experiments were run.
* surrogates: Filename to store surrogate models in or load surrogate models from.
* hyperparameters: Hyperparameters for which to find symbolic defaults, 
 must match column names in 'experiment_data'.
* ignore*: Columns in the 'experiment_data' to ignore.
* filters*:  Specify which experiments to use to build surrogate models.
 Expects a dict with 'hyperparameter: value' pairs, e.g. to use only rbf kernel results
 specify `{"svc__kernel": "rbf"}`.
* benchmark*: Configurations whose performance is also evaluated. Specified as dict
with as key the benchmark name, and as value the symbolic expression of the 
configuration. E.g.: `{"sklearn_scale": "make_tuple(1., truediv(1., mul(p, xvar)))"}`. 
