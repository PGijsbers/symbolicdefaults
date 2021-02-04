install_requirements = function() {
  install.packages("mlr")
  install.packages("OpenML")
  install.packages("farff")
  install.packages("data.table")
  install.packages("OpenML")
  install.packages("testthat")
  install.packages("mlrCPO")
  install.packages("jsonlite")
  install.packages("parallelMap")
  install.packages("reticulate")
  install.packages("BBmisc")
  install.packages("batchtools")
  install.packages("mlr3misc")
  install.packages("lgr")
  install.packages("e1071")
  install.packages("glmnet")
  install.packages("rpart")
  install.packages("xgboost")
  install.packages("ranger")
  install.packages("RcppHNSW")
  install.packages("snow")
}


# Iterate over a list converting character "TRUE", "FALSE" to logical.
parse_lgl = function(lst) {
  lst = lapply(lst, function(x) {
    if (!is.na(x)) {
      if (x == "FALSE" || x == "TRUE")
        x = as.logical(x)
    }
    return(x)
  })
  Filter(Negate(is.na), lst)
}

# Parse a tupel into a expression
# @param str [character]: tuple to parse
# @example
#  parse_tuple("make_tuple(add(a, b), 4.3)")
parse_tuple = function(str) {
  s = gsub("make_tuple", "list", str)
  parse(text=s)
}

get_problem_json = function(problem) {
  jsonlite::read_json(paste0("problems/", problem, ".json"))
}

get_deap_operations = function() {
  list(
    add = function(x, y) x + y,
    sub = function(x, y) x - y,
    mul = function(x, y) x * y,
    truediv = function(x, y) x / y,
    if_gt = function(a, b, c, d) if (a > b) c else d,
    pow = function(x, y) x^y,
    expit = function(x) exp(x),
    neg = function(x) -x,
    max = max,
    min = min,
    list = base::list
  )
}

# Read Metadata from JSON for a problem
read_metadata = function(problem) {
  setnames(fread(get_problem_json(problem)$metadata), "V1", "task_id")
}

# Read Experiment Data for a problem, return a data.table
read_expdata = function(problem) {
  farff::readARFF(get_problem_json(problem)$experiment_data)
}

# Fix task ids that have changed on OpenML since  the results were obtained.
fix_task = function(task) {
  if (task == 168759) task = 167211 # Satellite
  if (task == 168761) task = 168912 # sylvine
  if (task == 168770) task = 168909 # Dilbert
  return(task)
}


# Evaluate a tuple (see test) using metadata from a task and functions defined in
# get_deap_operations.
# Test:
# eval_tuple(str = "make_tuple(mcp, add(4,p))", "mlr_svm", 3)
# eval_tuple(str = "make_tuple(mcp, add(4,p),  3)", "mlr_rf", 3)
eval_tuple = function(problem, task, str) {
  if (problem == "mlr_random forest")
    problem = "mlr_rf"
  prob = get_problem_json(problem)
  # Parse formula
  symb = as.list(read_metadata(problem)[task_id == fix_task(task),])
  opts = get_deap_operations()
  lst = eval(parse_tuple(str), envir = as.environment(c(symb, opts)))
  # Get names and append filters / fixed
  hpnames = setdiff(names(prob$hyperparameters), c(names(prob$fixed), names(prob$filter)))
  if (length(lst) != length(hpnames)) stop("Hyperparameter names not equal to symbol length")
  names(lst) = hpnames
  c(lst, prob$filters, prob$fixed)
}


# Sanitize algorithm string
sanitize_algo = function(algo) {
  if (grepl("classif.", algo, fixed = TRUE))
    algo = algo
  else if (grepl("mlr_", algo, fixed = TRUE)) {
    algo = gsub("mlr_", "classif.", algo, fixed = TRUE)
  } else {
    algo = paste0("classif.", algo)
  }

  # Rename to mlr counterparts
  if (algo == "classif.rf" || algo == "classif.random forest") {
    mlr_algo_name = "classif.ranger"
  } else if (algo == "classif.knn") {
    mlr_algo_name = "classif.RcppHNSW"
  } else {
    mlr_algo_name = algo
  }
  return(mlr_algo_name)
}

repairPoints2 = function(ps, hpars) {
  library(mlr3misc)

  hpars_num = keep(hpars, is.numeric)
  # Current invalid strategy: Replace with 10^-6
  invalids = map_lgl(hpars_num, is.infinite) | map_lgl(hpars_num, is.nan) | map_lgl(hpars_num, function(x) if (is.numeric(x)) abs(x) >= 1.2676506e+30 else FALSE)
  if (any(invalids)) {
    hpars_num[invalids] = 10^-6
  }
  too_big_int = map_lgl(hpars_num, function(x) if (is.numeric(x)) abs(x) >= .Machine$integer.max else FALSE)
  if (any(too_big_int)) {
    hpars_num[too_big_int] = .Machine$integer.max - 1
  }
  hpars = BBmisc::insert(hpars, hpars_num)
  hpars = repairPoint(ps, hpars)
  setNames(mlr3misc::pmap(list(map(ps$pars, "type")[names(hpars)], hpars), function(type, par) {
    if (type == "integer" && !is.null(par)) par = round(par)
    # if (type == "integer" && is.null(par)) par = 10^6
    return(par)
  }), names(hpars))
  keep(hpars, Negate(is.null))
}

# Get task ids for a given problem.
get_task_ids = function(problem) {
  p = reticulate::import_from_path("src")
  p$problem$Problem(problem)$valid_tasks
}

# Run an algorithm
# @param algo :: algorithm name, e.g. classif.svm
# @param task :: task_id, e.g. 3
# @param str  :: tuple string, e.g. "make_tuple(1,1)"
# @example
# run_algo("mlr_svm", 3, "make_tuple(1,1)")
run_algo = function(problem, task, str, ..., parallel = 10L) {

   if (set_parallel_by_task(parallel, task) && problem != "mlr_xgboost") {
		  parallelMap::parallelStartMulticore(parallel, level = "mlr.resample")
    }
    on.exit(parallelMap::parallelStop())

    lgr = get_logger("eval_logger")$set_threshold("info")
    lgr$add_appender(lgr::AppenderFile$new("runs/mlr_evaluation_log.log"))
    lgr$info(sprintf("Evaluating %s|%s|%s", problem, task, str))

    # Get learner and hyperpars
    lrn = make_preproc_pipeline(problem)
	  hpars = eval_tuple(problem, task, str)

    # Repair hyperparams according to paramset before predicting
    ps = filterParams(getParamSet(lrn), names(hpars))
    hpars = parse_lgl(hpars)
    hpars = repairPoints2(ps, hpars[names(ps$pars)])
    lrn = setHyperPars(lrn, par.vals = hpars)

    if (problem == "mlr_xgboost") {
      lrn = setHyperPars(lrn, nthread = 1L)
      if (!("nrounds" %in% names(hpars))) {
        lrn = setHyperPars(lrn, nrounds = 10L)
      }
    }

    bmr = try({
        # Some task have gotten different ids
        task = fix_task(task)
	      omltsk = getOMLTask(task)
        # Hack away bugs / missing stuff in OpenML, stratified does not matter as splits are fixed anyway
        if (task %in% c(2073, 41, 145681)) omltsk$input$estimation.procedure$parameters$stratified_sampling = "false"
		    if (task %in% c(146212, 168329, 168330, 168331, 168332, 168339, 145681, 168331)) omltsk$input$evaluation.measures = ""
        z = convertOMLTaskToMlr(omltsk, measures = mmce)
        if (problem == "mlr_random forest") {
          nfeats = sum(z$mlr.task$task.desc$n.feat)
          if (task %in% c(3, 219, 15)) nfeats = 0.8*nfeats
          lrn = setHyperPars(lrn, mtry = max(min(hpars[["mtry"]], nfeats), 1))
        }
        if (problem == "mlr_knn") {
		      hpars[["M"]] = min(64, hpars[["M"]])
		      hpars[["ef_construction"]] = min(4096, hpars[["ef_construction"]])
	      }
	    lrn = setHyperPars(lrn)
		    benchmark(lrn, z$mlr.task, z$mlr.rin, measures = c(z$mlr.measures, list(mlr::logloss, mlr::mmce)))
		})
    aggr = bmr$results[[1]][[1]]$aggr
    measure = "logloss.test.mean"
    lgr$info(sprintf("Result: %s: %s", measure, aggr[[measure]]))
    bmr$results[[1]][[1]]
}

# Run an algorithm
# @param algo :: algorithm name, e.g. classif.svm
# @param task :: task_id, e.g. 3
# @param str  :: tuple string, e.g. "make_tuple(1,1)"
# @example
# run_algo("mlr_svm", 3, "make_tuple(1,1)")
run_algo_2 = function(problem, task, cfg, parallel = 10L) {

   if (set_parallel_by_task(parallel, task) && problem != "mlr_xgboost")
          parallelMap::parallelStartMulticore(parallel, level = "mlr.resample")
    on.exit(parallelMap::parallelStop())

    lgr = get_logger("eval_logger")$set_threshold("info")
    lgr$add_appender(lgr::AppenderFile$new("runs/mlr_evaluation_log.log"))

    lgr$info(sprintf("Evaluating %s|%s|%s", problem, task, paste0(as.character(cfg), collapse=",")))

    # Get learner and hyperpars
    lrn = make_preproc_pipeline(problem)
    hpars = cfg

    # Repair hyperparams according to paramset before predicting
    ps = filterParams(getParamSet(lrn), names(hpars))
    hpars = parse_lgl(hpars)
    hpars = repairPoints2(ps, hpars[names(ps$pars)])
    lrn = setHyperPars(lrn, par.vals = hpars)
    if (problem == "mlr_xgboost")
      lrn = setHyperPars(lrn, nthread = 1L)
    bmr = try({
        # Some task have gotten different ids
        task = fix_task(task)
        omltsk = getOMLTask(task)
        # Hack away bugs / missing stuff in OpenML, stratified does not matter as splits are fixed anyway
        if (task %in% c(2073, 41, 145681)) omltsk$input$estimation.procedure$parameters$stratified_sampling = "false"
        if (task %in% c(146212, 168329, 168330, 168331, 168332, 168339, 145681, 168331)) omltsk$input$evaluation.measures = ""
        z = convertOMLTaskToMlr(omltsk, measures = mmce)
        if (problem == "mlr_random forest") {
          nfeats = sum(z$mlr.task$task.desc$n.feat)
          if (task %in% c(3, 219, 15)) nfeats = 0.8*nfeats
          lrn = setHyperPars(lrn, mtry = max(min(hpars[["mtry"]], nfeats), 1))
        }
        if (problem == "mlr_knn") {
          hpars[["M"]] = min(64, hpars[["M"]])
          hpars[["ef_construction"]] = min(4096, hpars[["ef_construction"]])
        }
      lrn = setHyperPars(lrn)
        benchmark(lrn, z$mlr.task, z$mlr.rin, measures = c(z$mlr.measures, list(mlr::logloss, mlr::mmce)))
    })
    aggr = bmr$results[[1]][[1]]$aggr
    measure = "logloss.test.mean"
    lgr$info(sprintf("Result: %s: %s", measure, aggr[[measure]]))
    bmr$results[[1]][[1]]
}

# Write results from registry to a CSV
# @param pname: problem_name
# @param out_suffix: suffix for output file
problem_results_to_csv = function(pname, out_suffix) {
  jt = getJobTable()[, success := !is.na(done) & is.na(error)]
  jt = cbind(jt, setnames(map_dtr(jt$algo.pars, identity), "problem", "problem_name"))
  jt = jt[problem_name == pname, c("job.id", "problem_name", "task", "str", "success")]
  jt = merge(jt, imap_dtr(get_problem_json(pname)$benchmark, function(x, y) data.table("default" = y, "str" = x)), by = "str")
  if (nrow(jt[(!success)]))
    message("Unfinished jobs: ", paste0(jt[(!success)]$job.id, collapse = ","))
  jt = jt[(success)]
  jt = cbind(jt,
    map_dtr(reduceResultsList(jt$job.id, function(x)
      t(x$aggr[c("timetrain.test.sum", "timepredict.test.sum", "mmce.test.mean")])), data.table)
  )
  # Update column order
  jt = jt[, c(3, 4, 1, 6, 9, 7, 8, 2)]
  fwrite(jt, file = paste0("data/results_gecco/", pname, "_", out_suffix, ".csv"))
}

symbolic_results_to_csv = function(pname, out_suffix, default_name = "symbolic_default", jobs = NULL) {
  jt = getJobTable(jobs)[, success := !is.na(done) & is.na(error)]
  jt = cbind(jt, setnames(map_dtr(jt$algo.pars, identity), "problem", "problem_name"))
  jt = jt[problem_name == pname, c("job.id", "problem_name", "task", "str", "success")]
  if (nrow(jt[(!success)]))
    message("Unfinished jobs: ", paste0(jt[(!success)]$job.id, collapse = ","))
  jt = jt[(success)]
  jt = cbind(jt,
    map_dtr(reduceResultsList(jt$job.id, function(x)
      t(x$aggr[c("timetrain.test.sum", "timepredict.test.sum", "logloss.test.mean", "mmce.test.mean")])), data.table)
  )
  files = c(
    "constant" = paste0("data/generated_defaults/constants/",substr(pname, 5, 100),"_cst_found_by_mean_rank.csv"),
    "short" = paste0("data/generated_defaults/symbolic/",substr(pname, 5, 100),"_found_by_hp_based_max_length.csv"),
    "long" = paste0("data/generated_defaults/symbolic/",substr(pname, 5, 100),"_found_by_mean_rank.csv")
  )
  imap(files, function(x,n) {
    if (file.exists(x)) {
      df = fread(x)
      jt[jt$str %in% df$expression, default := n]
    }
  })
  # Update column order
  jt = jt[, c("problem_name","task","str","default", "logloss.test.mean", "mmce.test.mean","timetrain.test.sum","timepredict.test.sum","job.id")]
  fwrite(jt, file = paste0("data/results_gecco/", pname, "_", out_suffix, ".csv"))
}

problem_nn_results_to_csv = function(pname, out_suffix) {
  jt = getJobTable()[, success := !is.na(done) & is.na(error)]
  jt = cbind(jt, setnames(map_dtr(jt$algo.pars, function(x) {
    y = as.data.table(x[c("task", "problem")])
    y$str = stringify_list(x["cfg"])
    return(y)
    }), "problem", "problem_name"))
  jt = jt[problem_name == pname, c("job.id", "problem_name", "task", "str", "success")]
  # jt = merge(jt, imap_dtr(get_problem_json(pname)$benchmark, function(x, y) data.table("default" = y, "str" = x)), by = "str")
  if (nrow(jt[(!success)]))
    message("Unfinished jobs: ", paste0(jt[(!success)]$job.id, collapse = ","))
  jt = jt[(success)]
  jt = cbind(jt,
    map_dtr(reduceResultsList(jt$job.id, function(x)
      t(x$aggr[c("timetrain.test.sum", "timepredict.test.sum", "mmce.test.mean", "logloss.test.mean")])), data.table)
  )
  jt$default = "nearest_neighbour"
  # Update column order
  jt = jt[, c("problem_name","task","str", "default", "logloss.test.mean", "mmce.test.mean","timetrain.test.sum","timepredict.test.sum","job.id")]
  fwrite(jt, file = paste0("data/results_gecco/", pname, "_", out_suffix, ".csv"))
}


stringify_list = function(x) {paste0(imap_chr(x$cfg, function(x,n) {paste0(x, ":", n)}), collapse=",")}

filter_run_files = function(jobs, run_files) {
  dd = rbindlist(map(run_files, fread))
  names(dd) = c("problem", "task", "str")
  dd$problem = paste0("mlr_", dd$problem)
  jt = getJobTable(jobs)
  d2 = map_dtr(jt$algo.pars, identity)
  d2$job.id = jt$job.id
  unique(d2[dd, on = names(dd), nomatch = 0]$job.id)
}