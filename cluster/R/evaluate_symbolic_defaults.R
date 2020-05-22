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
source_packages = c("mlr", "mlrCPO", "OpenML", "jsonlite", "data.table", "parallelMap", "lgr", "mlr3misc")

# Create Job Registry
if (!file.exists(REG_DIR)) {
  reg = makeExperimentRegistry(
    file.dir = REG_DIR,
    seed = 1,
    packages = source_packages,
    source = source_files
  )
  addProblem("symbolic_best_best")
  addAlgorithm("run_algo", fun = function(data, job, instance, ...) {run_algo(..., parallel = RESAMPLE_PARALLEL_CPUS)})

  # Each line in grd is a configuration
  grd = fread("data/random_search_30k.csv")
  grd = grd[, c("problem", "task", "expression")]
  grd[problem == "random forest", ]$problem = "rf"
  grd[, str := expression][, expression := NULL][, problem := paste0("mlr_", problem)]
  grd = unique(grd)
  addExperiments(algo.designs = list(run_algo = grd))

  grd = fread("data/random_search_30k_xgb.csv")
  grd = grd[, c("problem", "task", "expression")]
  grd[problem == "random forest", ]$problem = "rf"
  grd[, str := expression][, expression := NULL][, problem := paste0("mlr_", problem)]
  grd = unique(grd)
  addExperiments(algo.designs = list(run_algo = grd))

} else {
  reg = loadRegistry(REG_DIR, writeable = TRUE)
}

reg$cluster.functions = makeClusterFunctionsSocket(6)



# Submit SVM jobs ### ssh: christoph
jobs = findNotDone()$job.id
while (length(jobs)) {
  jobs = setdiff(findNotDone()$job.id, findRunning()$job.id)
  if (length(jobs)) {
    jt = getJobTable(jobs)
    jt = cbind(jt, setnames(map_dtr(jt$algo.pars, identity), "problem", "problem_name"))
    jobs = intersect(jobs, jt[problem_name %in% c("mlr_svm", "mlr_glmnet", "mlr_xgboost"), ]$job.id)
    try({submitJobs(sample(jobs))})
  }
  Sys.sleep(3)
}

# Submit xgboost jobs ### ssh: compstat
jobs = findNotDone()$job.id
while (length(jobs)) {
  jobs = setdiff(findNotDone()$job.id, findRunning()$job.id)
  if (length(jobs)) {
    jt = getJobTable(jobs)
    jt = cbind(jt, setnames(map_dtr(jt$algo.pars, identity), "problem", "problem_name"))
    jobs = intersect(jobs, jt[problem_name %in% c("mlr_rpart", "mlr_rf", "mlr_knn"), ]$job.id)
    try({submitJobs(sample(jobs))})
  }
  Sys.sleep(3)
}



# Reduce results.
if (FALSE) {
  reg = loadRegistry(REG_DIR, writeable = FALSE)                                                    # STATUS / PLANNING
  problem_results_to_csv(pname = "mlr_rpart", out_suffix = "real_data_baselines_results")           # done
  problem_results_to_csv(pname = "mlr_svm", out_suffix = "real_data_baselines_results")             # running: ssh:christoph
  problem_results_to_csv(pname = "mlr_glmnet", out_suffix = "real_data_baselines_results")          # done
  problem_results_to_csv(pname = "mlr_rf", out_suffix = "real_data_baselines_results")              # done
  problem_results_to_csv(pname = "mlr_xgboost", out_suffix = "real_data_baselines_results")         # missing: run on: ssh:compstat
  problem_results_to_csv(pname = "mlr_knn", out_suffix = "real_data_baselines_results")             # done
} else if (REG_DIR == "registry_symbolics") {
  reg = loadRegistry(REG_DIR, writeable = FALSE)                                                    # STATUS / PLANNING
  symbolic_results_to_csv(pname = "mlr_rpart", out_suffix = "real_data_symbolic_results")           # done
  symbolic_results_to_csv(pname = "mlr_svm", out_suffix = "real_data_symbolic_results")             # running: ssh:christoph
  symbolic_results_to_csv(pname = "mlr_glmnet", out_suffix = "real_data_symbolic_results")          # done
  symbolic_results_to_csv(pname = "mlr_rf", out_suffix = "real_data_symbolic_results")              # done
  symbolic_results_to_csv(pname = "mlr_xgboost", out_suffix = "real_data_symbolic_results")         # missing: run on: ssh:compstat
  symbolic_results_to_csv(pname = "mlr_knn", out_suffix = "real_data_symbolic_results")             # done
}

}

  jt = getJobTable()
  jt = cbind(jt, setnames(map_dtr(jt$algo.pars, identity), "problem", "problem_name"))
  jobs = intersect(jobs, jt[problem_name %in% c("mlr_xgboost"), ]$job.id)
  try({submitJobs(sample(jobs))})


