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

source_files = c("cluster/R/CPO_maxfact.R", "cluster/R/RLearner_classif_rcpphnsw.R", "cluster/R/helpers.R", "cluster/R/config.R")
sapply(source_files, source)
source_packages = c("mlr", "mlrCPO", "OpenML", "jsonlite", "data.table", "parallelMap", "lgr")


jobs = c("mlr_svm", "mlr_rpart", "mlr_rf", "mlr_knn", "mlr_glmnet", "mlr_xgboost")

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
  for (job in jobs) {
    benchmarks = get_problem_json(job)$benchmark
    tasks = get_task_ids(job)
    grd = CJ(problem = job, task = tasks, str = unlist(benchmarks))
    addExperiments(algo.designs = list(run_algo = grd))
  }
} else {
  reg = loadRegistry(REG_DIR, writeable = TRUE)
}

reg$cluster.functions = makeClusterFunctionsSocket(6)

while (TRUE) {
  jobs = setdiff(findNotDone()$job.id, findRunning()$job.id)[1:10]
  try({submitJobs(jobs)})
  Sys.sleep(3)
}

sapply(source_packages, install.packages)