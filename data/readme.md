# Data Files
**adaboost.arff**: results of random search with the AdaBoostClassifier of 500 iterations over 100 tasks.

**svc.arff**: results of random search with the SupportVectorClassifier of 500 iterations over 100 tasks.

**rf.arff**: results of random search with the RandomForestClassifier of 500 iterations over 94 tasks.

**metadata.csv**: the original file with metafeatures describing the datasets of 100 tasks.

**before_pipeline_metafeatures.csv**: metafeatures derived from `metadata.csv`, describing the datasets of 100 tasks. Specifically:

 - n: number of instances
 - p: number of features
 - m: number of classes
 - mcp: ratio of majority class
 - mkd: median kernel distance as computed by kernlab::digest
 - rc: ratio of categorical features
 - xvar: `X.var()` (added because it is used in the scikit-learn default for SVC.gamma)
 
 These metafeatures are calculated on the data *before* any of the transformation occur in the pipeline (e.g. imputation, scaling, encoding).
 
 **after_pipeline_metafeatures.csv**: The same metafeatures as in `before_pipeline_metafeatures.csv` but calculated on the data transformed by the preprocessing pipeline.
 
 **{classifier}_surrogates.pkl**: 
 Pickle blob with a surrogate model for each task (`Dict[task: int, surrogate: object]`).
 