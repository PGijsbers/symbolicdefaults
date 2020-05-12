library(mlr)
library(OpenML)
library(farff)
library(data.table)
library(OpenML)
library(testthat)
library(checkmate)
library(mlr)
library(mlrCPO)
library(jsonlite)
library(parallelMap)
library(lgr)

source("cluster/R/CPO_maxfact.R")
source("cluster/R/RLearner_classif_rcpphnsw.R")
source("cluster/R/helpers.R")

lgr = get_logger("eval_logger")$set_threshold("info")
lgr$add_appender(lgr::AppenderFile$new("runs/mlr_evaluation_log.log"))

# Run an algorithm
# @param algo :: algorithm name, e.g. classif.svm
# @param task :: task_id, e.g. 3
# @param str  :: tuple string, e.g. "make_tuple(1,1)"
# @example
# run_algo("mlr_svm", 3, "make_tuple(1,1)")
run_algo = function(problem, task, str) {
  lgr$info(sprintf("Evaluating %s  on task %s for %s", problem, task, str))
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
}

parallelStartMulticore(5, level = "mlr.resample")
a = run_algo("mlr_svm", 3, "make_tuple(1,1)")
