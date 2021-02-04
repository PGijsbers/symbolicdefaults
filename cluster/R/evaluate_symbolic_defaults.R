# install_requirements()
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
library(lgr)

REG_DIR = "cluster/registry_symbolics"
source_files = c("cluster/R/CPO_maxfact.R", "cluster/R/RLearner_classif_rcpphnsw.R", "cluster/R/helpers.R", "cluster/R/config.R")
sapply(source_files, source)
source_packages = c("mlr", "mlrCPO", "OpenML", "jsonlite", "data.table", "parallelMap", "lgr", "mlr3misc")

ALGOS = "mlr_svm"



####################################################################################################
### Results: unitmp

run_files = list.files("data/generated_defaults/symbolic", full.names=TRUE)

# Create Job Registry
if (!file.exists(REG_DIR)) {
  reg = makeExperimentRegistry(
    file.dir = REG_DIR,
    seed = 1,
    packages = source_packages,
    source = source_files
  )
  addProblem("symbolic_best_best")
  addAlgorithm("run_algo", fun = function(data, job, instance, ...) {run_algo(..., parallel = 1)})

  # Each line in grd is a configuration
  for (file in run_files) {
    grd = fread(file)
    grd = grd[, c("algorithm", "task", "expression")]
    grd[algorithm == "random forest", ]$problem = "rf"
    grd[, str := expression][, expression := NULL][, problem := paste0("mlr_", algorithm)][, algorithm := NULL]
    grd = unique(grd)
    addExperiments(algo.designs = list(run_algo = grd))
  }

} else {
  reg = loadRegistry(REG_DIR, writeable = TRUE)
  # unlink(REG_DIR, TRUE)
}


reg$cluster.functions = makeClusterFunctionsSocket(24)

# Submit jobs
jobs = setdiff(findNotDone()$job.id, findRunning()$job.id)
jobs = filter_run_files(jobs)
while (length(jobs)) {
  if (length(jobs)) {
    jt = getJobTable(jobs)
    jt = cbind(jt, setnames(map_dtr(jt$algo.pars, identity), "problem", "problem_name"))
    jt = jt[problem_name %in% ALGOS]
    try({submitJobs(sample(jt$job.id))})
  }
  Sys.sleep(500)
}



jobs = findDone()
jobs = filter_run_files(jobs, run_files)
reg = loadRegistry(REG_DIR, writeable = FALSE)                                                      # STATUS / PLANNING
#symbolic_results_to_csv(pname = "mlr_rpart", out_suffix = "real_data_symbolic_results_2")           # done
symbolic_results_to_csv(pname = "mlr_svm", out_suffix = "real_data_symbolic_results_2", jobs = jobs)             # running: ssh:christoph
#symbolic_results_to_csv(pname = "mlr_glmnet", out_suffix = "real_data_symbolic_results_2")          # done
#symbolic_results_to_csv(pname = "mlr_rf", out_suffix = "real_data_symbolic_results_2")              # done
#symbolic_results_to_csv(pname = "mlr_xgboost", out_suffix = "real_data_symbolic_results_2")         # missing: run on: ssh:compstat
#symbolic_results_to_csv(pname = "mlr_knn", out_suffix = "real_data_symbolic_results_2")             # done





####################################################################################################
### Baselines: uni2
run_files = c(
  "data/svm_nearest_neighbors.csv",
  "data/rpart_nearest_neighbors.csv",
  "data/knn_nearest_neighbors.csv",
  "data/glmnet_nearest_neighbors.csv",
  "data/rf_nearest_neighbors.csv",
  "data/xgboost_nearest_neighbors.csv"
)

# Create Job Registry
if (!file.exists(REG_DIR)) {
  reg = makeExperimentRegistry(
    file.dir = REG_DIR,
    seed = 1,
    packages = source_packages,
    source = source_files
  )
  addProblem("symbolic_best_best")
  addAlgorithm("run_algo_2", fun = function(data, job, instance, ...) {run_algo_2(..., parallel = RESAMPLE_PARALLEL_CPUS)})

  # Each line in grd is a configuration
  for (file in run_files) {
    grd = fread(file)
    grd = data.table(task = grd$V1, cfg = pmap(grd[, 2:ncol(grd)], list), algorithm = gsub("_nearest_neighbors.csv", "", gsub("data/", "", file)))
    grd = grd[, c("algorithm", "task", "cfg")]
    grd[, problem := paste0("mlr_", algorithm)][, algorithm := NULL]
    grd = unique.data.frame(grd)
    addExperiments(algo.designs = list(run_algo_2 = grd))
  }
} else {
  reg = loadRegistry(REG_DIR, writeable = TRUE)
  # unlink(REG_DIR, TRUE)
}
reg$cluster.functions = makeClusterFunctionsSocket(24)


# Submit jobs
ALGOS = c("mlr_svm", "mlr_rpart")

jobs = findNotDone()$job.id
while (length(jobs)) {
  jobs = setdiff(findNotDone()$job.id, findRunning()$job.id)
  if (length(jobs)) {
    jt = getJobTable(jobs)
    jt$problem_name = map_chr(jt$algo.pars, function(x) x$problem)
    jt = jt[problem_name %in% ALGOS]
    try({submitJobs(sample(jt$job.id))})
  }
  Sys.sleep(500)
}

reg = loadRegistry(REG_DIR, writeable = FALSE)                                                    # STATUS / PLANNING
#problem_nn_results_to_csv(pname = "mlr_rpart", out_suffix = "real_data_nn_results")
problem_nn_results_to_csv(pname = "mlr_svm", out_suffix = "real_data_nn_results")
#problem_nn_results_to_csv(pname = "mlr_glmnet", out_suffix = "real_data_nn_results")
#problem_nn_results_to_csv(pname = "mlr_rf", out_suffix = "real_data_nn_results")
#problem_nn_results_to_csv(pname = "mlr_xgboost", out_suffix = "real_data_nn_results")
#problem_nn_results_to_csv(pname = "mlr_knn", out_suffix = "real_data_nn_results")

####################################################################################################
### Implementation defaults uni3

run_files = c(
  "data/mlr_rf_real_data_baselines_results.csv",
  "data/mlr_svm_real_data_baselines_results.csv",
  "data/mlr_glmnet_real_data_baselines_results.csv",
  "data/mlr_xgboost_real_data_baselines_results.csv",
  "data/mlr_rpart_real_data_baselines_results.csv",
  "data/mlr_knn_real_data_baselines_results.csv"
)

# Create Job Registry
if (!file.exists(REG_DIR)) {
  reg = makeExperimentRegistry(
    file.dir = REG_DIR,
    seed = 1,
    packages = source_packages,
    source = source_files
  )
  addProblem("symbolic_best_best")
  addAlgorithm("run_algo", fun = function(data, job, instance, ...) {run_algo(..., parallel = 1)})

  # Each line in grd is a configuration
  for (file in run_files) {
    grd = fread(file)
    grd = grd[, c("problem_name", "task", "str", "default")]
    grd = unique(grd)
    grd[, problem := grd$problem_name][, problem_name := NULL]
    # Do not run excluded tasks
    exclude = unlist(jsonlite::read_json(paste0("problems/", unique(grd$problem), ".json"))$exclude)
    grd = grd[!(grd$task %in% exclude), ]
    addExperiments(algo.designs = list(run_algo = grd))
  }

} else {
  reg = loadRegistry(REG_DIR, writeable = TRUE)
  # unlink(REG_DIR, TRUE)
}
reg$cluster.functions = makeClusterFunctionsSocket(24)
# Submit jobs
jobs = findNotDone()$job.id
while (length(jobs)) {
  jobs = setdiff(findNotDone()$job.id, findRunning()$job.id)
  if (length(jobs)) {
    jt = getJobTable(jobs)
    jt = cbind(jt, setnames(map_dtr(jt$algo.pars, identity), "problem", "problem_name"))
    # jt = jt[problem_name %in% ALGOS]

    try({submitJobs(sample(jt$job.id))})
  }
  Sys.sleep(500)
}




# Collect results.
# if (FALSE) {
#   reg = loadRegistry(REG_DIR, writeable = FALSE)                                                    # STATUS / PLANNING
#   problem_results_to_csv(pname = "mlr_rpart", out_suffix = "real_data_baselines_results")           # done
#   problem_results_to_csv(pname = "mlr_svm", out_suffix = "real_data_baselines_results")             # running: ssh:christoph
#   problem_results_to_csv(pname = "mlr_glmnet", out_suffix = "real_data_baselines_results")          # done
#   problem_results_to_csv(pname = "mlr_rf", out_suffix = "real_data_baselines_results")              # done
#   problem_results_to_csv(pname = "mlr_xgboost", out_suffix = "real_data_baselines_results")         # missing: run on: ssh:compstat
#   problem_results_to_csv(pname = "mlr_knn", out_suffix = "real_data_baselines_results")             # done
# } else if (REG_DIR == "registry_symbolics") {
  reg = loadRegistry(REG_DIR, writeable = FALSE)                                                    # STATUS / PLANNING
  symbolic_results_to_csv(pname = "mlr_rpart", out_suffix = "real_data_symbolic_results_2")           # done
  symbolic_results_to_csv(pname = "mlr_svm", out_suffix = "real_data_symbolic_results_2")             # running: ssh:christoph
  symbolic_results_to_csv(pname = "mlr_glmnet", out_suffix = "real_data_symbolic_results_2")          # done
  symbolic_results_to_csv(pname = "mlr_rf", out_suffix = "real_data_symbolic_results_2")              # done
  symbolic_results_to_csv(pname = "mlr_xgboost", out_suffix = "real_data_symbolic_results_2")         # missing: run on: ssh:compstat
  symbolic_results_to_csv(pname = "mlr_knn", out_suffix = "real_data_symbolic_results_2")             # done
}

jt = getJobTable()
jt = cbind(jt, setnames(map_dtr(jt$algo.pars, identity), "problem", "problem_name"))
jobs = intersect(jobs, jt[problem_name %in% c("mlr_xgboost"), ]$job.id)
try({submitJobs(sample(jobs))})
