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



# Run an algorithm
# @param algo :: algorithm name, e.g. classif.svm
# @param task :: task_id, e.g. 3
# @param str  :: tuple string, e.g. "make_tuple(1,1)"
# @example
# run_algo("mlr_svm", 3, "make_tuple(1,1)")
run_algo = function(problem, task, str, parallel = 10L) {
  if (parallel)
    parallelMap::parallelStartMulticore(parallel, level = "mlr.resample")
    on.exit(parallelMap::parallelStop())

  lgr = get_logger("eval_logger")$set_threshold("info")
  lgr$add_appender(lgr::AppenderFile$new("runs/mlr_evaluation_log.log"))
  lgr$info(sprintf("Evaluating %s|%s|%s", problem, task, str))
  lrn = make_preproc_pipeline(problem)
  hpars = eval_tuple(problem, task, str)
  setHyperPars(lrn, par.vals = parse_lgl(hpars))
  bmr = try({
    omltsk = getOMLTask(task)
    z = convertOMLTaskToMlr(omltsk, measures = mmce)
    benchmark(lrn, z$mlr.task, z$mlr.rin, measures = z$mlr.measures)
  })
  aggr = bmr$results[[1]][[1]]$aggr
  measure = "mmce.test.mean"
  lgr$info(sprintf("Result: %s: %s", measure, aggr[[measure]]))
  bmr$results[[1]][[1]]
}

jobs = c("mlr_svm", "mlr_rpart", "mlr_rf", "mlr_knn", "mlr_glmnet", "mlr_xgboost")

# Submit Jobs
if (!file.exists(REG_DIR)) {
  reg = makeExperimentRegistry(
    file.dir = REG_DIR,
    seed = 1,
    conf.file = "cluster/R/batchtools.conf.R",
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

submitJobs(1)