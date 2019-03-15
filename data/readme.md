# Data Files
**adaboost.arff**: results of random search with the AdaBoostClassifier of 500 iterations over 100 tasks.

**svc.arff**: results of random search with the SupportVectorClassifier of 500 iterations over 100 tasks.

**metadata.csv**: metafeatures describing the datasets of 100 tasks.

**pp_metadata.csv**: metafeatures derived from `metadata.csv`, describing the datasets of 100 tasks. Specifically:

 - n: number of instances
 - p: number of features
 - m: number of classes
 - mcp: ratio of majority class
 - mkd: median kernel distance as computed by kernlab::digest
 - rc: ratio of categorical features
 
 **{classifier}_surrogates.pkl**: 
 Pickle blob with a surrogate model for each task (`Dict[task: int, surrogate: object]`).
 