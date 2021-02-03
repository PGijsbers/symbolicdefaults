library(batchtools)
library(data.table)
source_files = c("cluster/R/helpers.R")
sapply(source_files, source)
source_packages = c("mlr", "mlrCPO", "OpenML", "jsonlite", "data.table", "parallelMap", "lgr", "mlr3misc")


REG_DIR = "xgboost_runs"

submit_py = function(problem, task, algo) {
    alg_short = strsplit(algo, " ")[[1]][2]
    job_name = paste0("jobs/", problem, "_", alg_short, "_", task, ".job")
    outdir = paste0("results/", problem, "_", alg_short, "/")
    search_string = paste0("python3.6 src/main.py mlr_", problem, " -o ", outdir," ",algo, " -t ", task)
    if (algo == "-a random_search")
      search_string = paste0(search_string, ' -mss 3')
    system(search_string)
}

# Create Job Registry
if (!file.exists(REG_DIR)) {
  reg = makeExperimentRegistry(
    file.dir = REG_DIR,
    seed = 1
  )
  addProblem("runjob")
  addAlgorithm("runalgo", fun = function(data, job, instance, ...) {submit_py(...)})

  job = "xgboost"
  for (job in jobs) {
    tasks =  get_task_ids(paste0("mlr_", job))
    grd = CJ(problem = job, task = tasks,  algo = c("-a random_search", "-a mupluslambda", "-cst True"))
    addExperiments(algo.designs = list(runalgo = grd))
  }
} else {
  reg = loadRegistry(REG_DIR, writeable = TRUE)
}

reg$cluster.functions = makeClusterFunctionsMulticore(2)
submitJobs(2:300)

# unlink(REG_DIR, TRUE, TRUE)