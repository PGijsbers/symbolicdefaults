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

read_metadata = function(problem) {
  setnames(fread(get_problem_json(problem)$metadata), "V1", "task_id")
}

read_expdata = function(problem) {
  farff::readARFF(get_problem_json(problem)$experiment_data)
}

fix_task = function(task) {
  if (task == 168759) task = 167211 # Satellite
  if (task == 168761) task = 168912 # sylvine
  if (task == 168770) task = 168909 # Dilbert
  return(task)
}

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


sanitize_algo = function(algo) {
  if (grepl("classif.", algo, fixed = TRUE))
    return(algo)
  else
    gsub("mlr_", "classif.", algo, fixed = TRUE)
}

get_task_ids = function(problem) {
  p = import_from_path("src.problem")
  p$Problem(problem)$valid_tasks
}

set_parallel_by_task = function(parallel, task) {
 	if (task %in% NO_PARALLEL_TASKS) parallel = 5
	return(parallel)
}


# Run an algorithm
# @param algo :: algorithm name, e.g. classif.svm
# @param task :: task_id, e.g. 3
# @param str  :: tuple string, e.g. "make_tuple(1,1)"
# @example
# run_algo("mlr_svm", 3, "make_tuple(1,1)")
run_algo = function(problem, task, str, parallel = 10L) {
   if (set_parallel_by_task(parallel, task))
		      parallelMap::parallelStartMulticore(parallel, level = "mlr.resample")
    on.exit(parallelMap::parallelStop())

    lgr = get_logger("eval_logger")$set_threshold("info")
    lgr$add_appender(lgr::AppenderFile$new("runs/mlr_evaluation_log.log"))
    lgr$info(sprintf("Evaluating %s|%s|%s", problem, task, str))

    lrn = make_preproc_pipeline(problem)
	  hpars = eval_tuple(problem, task, str)
    # Repair hyperparams according to paramset before predicting
    ps = filterParams(getParamSet(lrn), names(hpars))
    hpars = parse_lgl(hpars)
    hpars = repairPoint(ps, hpars[names(ps$pars)])
	  setHyperPars(lrn, par.vals = hpars)
    bmr = try({
        # Some task have gotten different ids
        task = fix_task(task)
	      omltsk = getOMLTask(task)
        # Hack away bugs / missing stuff in OpenML, stratified does not matter as splits are fixed anyway
        if (task %in% c(2073, 41, 145681)) omltsk$input$estimation.procedure$parameters$stratified_sampling = "false"
		    if (task %in% c(146212, 168329, 168330, 168331, 168332, 168339, 145681)) omltsk$input$evaluation.measures = ""
        z = convertOMLTaskToMlr(omltsk, measures = mmce)
		    benchmark(lrn, z$mlr.task, z$mlr.rin, measures = z$mlr.measures)
		})
    aggr = bmr$results[[1]][[1]]$aggr
    measure = "mmce.test.mean"
    lgr$info(sprintf("Result: %s: %s", measure, aggr[[measure]]))
    bmr$results[[1]][[1]]
}
