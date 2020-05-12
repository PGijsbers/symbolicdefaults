
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

# Test:
# eval_tuple(str = "make_tuple(mcp, add(4,p))", "mlr_svm", 3)
# eval_tuple(str = "make_tuple(mcp, add(4,p),  3)", "mlr_rf", 3)
eval_tuple = function(problem, task, str) {
  prob = get_problem_json(problem)
  # Parse formula
  symb = as.list(read_metadata(problem)[task_id == task,])
  opts = get_deap_operations()
  lst = eval(parse_tuple(str), envir = as.environment(c(symb, opts)))
  # Get names and append filters / fixed
  hpnames = setdiff(names(prob$hyperparameters), names(prob$fixed))
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