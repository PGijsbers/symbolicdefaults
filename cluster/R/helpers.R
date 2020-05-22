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
    algo = return(algo)
  else if (grepl("mlr_", algo, fixed = TRUE)) {
    algo = gsub("mlr_", "classif.", algo, fixed = TRUE)
  } else {
    algo = paste0("classif.", algo)
  }

  # Rename to mlr counterparts
  if (algo == "classif.rf") {
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

  # Current invalid strategy: Replace with 10^-6
  invalids = map_lgl(hpars, is.infinite) | map_lgl(hpars, is.nan)
  if (any(invalids)) {
    hpars[invalids] = 10^-6
  }
  hpars = repairPoint(ps, hpars)
  setNames(mlr3misc::pmap(list(map(ps$pars, "type")[names(hpars)], hpars), function(type, par) {
    if(type == "integer") par = round(par)
    return(par)
  }), names(hpars))
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
run_algo = function(problem, task, str, parallel = 10L) {

   if (set_parallel_by_task(parallel, task) && problem != "mlr_xgboost")
		      parallelMap::parallelStartMulticore(parallel, level = "mlr.resample")
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
		    benchmark(lrn, z$mlr.task, z$mlr.rin, measures = z$mlr.measures)
		})
    aggr = bmr$results[[1]][[1]]$aggr
    measure = "mmce.test.mean"
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
  fwrite(jt, file = paste0("data/", pname, "_", out_suffix, ".csv"))
}

symbolic_results_to_csv = function(pname, out_suffix, default_name = "symbolic_default") {
  jt = getJobTable()[, success := !is.na(done) & is.na(error)]
  jt = cbind(jt, setnames(map_dtr(jt$algo.pars, identity), "problem", "problem_name"))
  jt = jt[problem_name == pname, c("job.id", "problem_name", "task", "str", "success")]
  # Each line in grd is a configuration
  grd = fread("data/random_search_30k.csv")
  grd = grd[, c("problem", "task", "expression")]
  grd[problem == "random forest", ]$problem = "rf"
  grd[, str := expression][, expression := NULL][, problem := paste0("mlr_", problem)]
  grd = unique(grd)[problem == pname, ]
  jt = merge(jt, unique(grd), by = c("str", "task"))
  if (nrow(jt[(!success)]))
    message("Unfinished jobs: ", paste0(jt[(!success)]$job.id, collapse = ","))
  jt = jt[(success)]
  jt = cbind(jt,
    map_dtr(reduceResultsList(jt$job.id, function(x)
      t(x$aggr[c("timetrain.test.sum", "timepredict.test.sum", "mmce.test.mean")])), data.table)
  )
  jt$default = default_name
  # Update column order
  jt = jt[, c("problem_name","task","str","default","mmce.test.mean","timetrain.test.sum","timepredict.test.sum","job.id")]
  fwrite(jt, file = paste0("data/", pname, "_", out_suffix, ".csv"))
}
