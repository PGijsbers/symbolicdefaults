library(mlr)
library(OpenML)
library(farff)
library(data.table)
library(OpenML)
library(testthat)
library(mlrCPO)
library(jsonlite)
library(parallelMap)
library(reticulate)
library(BBmisc)
library(batchtools)
library(mlr3misc)


source_files = c("cluster/R/CPO_maxfact.R", "cluster/R/RLearner_classif_rcpphnsw.R", "cluster/R/helpers.R", "cluster/R/config.R")
sapply(source_files, source)
source_packages = c("mlr", "mlrCPO", "OpenML", "jsonlite", "data.table", "parallelMap", "lgr")

# Create Job Registry
if (!file.exists(REG_DIR)) {
  reg = makeExperimentRegistry(
    file.dir = REG_DIR,
    seed = 1,
    packages = source_packages,
    source = source_files
  )
  addProblem("package_defaults")
  addAlgorithm("run_algo", fun = function(data, job, instance, ...) {run_algo(..., parallel = RESAMPLE_PARALLEL_CPUS)})

  jobs = c("mlr_svm", "mlr_rpart", "mlr_rf", "mlr_knn", "mlr_glmnet", "mlr_xgboost")

  jobs = "mlr_rpart"
  for (job in jobs) {
    benchmarks = get_problem_json(job)$benchmark
    tasks = 168332# get_task_ids(job)
    grd = CJ(problem = job, task = tasks, str = unlist(benchmarks))
    addExperiments(algo.designs = list(run_algo = grd))
  }
} else {
  reg = loadRegistry(REG_DIR, writeable = TRUE)
}

reg$cluster.functions = makeClusterFunctionsSocket(6)



jobs = findNotDone()$job.id
while (length(jobs)) {
  jobs = setdiff(findNotDone()$job.id, findRunning()$job.id)
  jt = getJobTable(jobs)
  jt = cbind(jt, setnames(map_dtr(jt$algo.pars, identity), "problem", "problem_name"))
  jobs = intersect(jobs, jt[problem_name %in% c("mlr_rpart","mlr_svm", "mlr_glmnet"), ]$job.id)
  try({submitJobs(jobs)})
  Sys.sleep(3)
}

jobs = findNotDone()$job.id
while (length(jobs)) {
  jobs = setdiff(findNotDone()$job.id, findRunning()$job.id)
  jt = getJobTable(jobs)
  jt = cbind(jt, setnames(map_dtr(jt$algo.pars, identity), "problem", "problem_name"))
  try({submitJobs(sample(jobs))})
  Sys.sleep(3)
}


# Reduce results.
if (FALSE) {
  reg = loadRegistry(REG_DIR, writeable = FALSE)
  problem_results_to_csv(pname = "mlr_rpart", out_suffix = "real_data_baselines_results")
  problem_results_to_csv(pname = "mlr_svm", out_suffix = "real_data_baselines_results")
  problem_results_to_csv(pname = "mlr_glmnet", out_suffix = "real_data_baselines_results")
}
