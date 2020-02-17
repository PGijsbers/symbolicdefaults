## Problem Definitions

A problem is defined by the hyperparameters to find symbolic defaults for, and the
experiment data to create surrogate models with.
This information is stored in JSON format.
The following fields are accepted (optional fields indicated with an asterisk (*)):

* name: recognizable name for the problem, used from the command line.
* rs_data: the file with experiment data from random search.
* metadata: file with metadata for each task on which experiments were run.
* surrogates: filename to store surrogate models in or load surrogate models from.
* hyperparameters: Hyperparameters for which to find symbolic defaults, 
 must match column names in 'rs_data'.
* default_filters*:  Use only configurations which match the filters.
* operators*: specifies which operators are allowed in the symbolic expressions. 
 Currently not used.
* checks*: A list of configurations to compare the found symbolic defaults to.

```
{
    "name": "svc_rbf",
    "rs_data": "data/svc_big.arff",
    "metadata": "data/after_pipeline_metafeatures.csv",
    "surrogates": "data/svc_surrogates_rbf_big.pkl",
    "hyperparameters": [
      "svc__C",
      "svc__gamma"
    ],
    "defaults_filters": {
      "svc__kernel": "rbf"
    },
    "operators": "default",
    "checks": {
      "sklearn_scale":"make_tuple(1., truediv(1., mul(p, xvar)))",
      "symbolic_best":"make_tuple(16., truediv(mkd, xvar))"
    }
}
```