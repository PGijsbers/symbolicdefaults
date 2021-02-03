setOMLConfig(cachedir = "../oml_cache/")

learner_packages = c("rpart", "glmnet" ,"xgboost", "e1071", "ranger", "RcppHNSW")

# The following tasks should not be computed in parallel;
# set_parallel_by_task (below) is used to check this.
NO_PARALLEL_TASKS = c(168329, 168338, 168332,9977, 3903, 146821)

# sets parallel to 0 for 'NO_PARALLEL_TASKS'
set_parallel_by_task = function(parallel, task) {
  if (task %in% c(NO_PARALLEL_TASKS, sapply(NO_PARALLEL_TASKS, fix_task))) parallel = 2
  return(parallel)
}


MTRY_FEATS=2