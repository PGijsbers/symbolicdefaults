# Preprocessing 0perator: Sets a maximum number of allowed factor levels, collapses all others to "collapsed"
cpoMaxFact <- makeCPO("max.fact",
  pSS(max.fact.no = .Machine$integer.max: integer[1, ]),
  fix.factors = TRUE,
  dataformat = "factor",
  cpo.train = {
    sapply(data, function(d) {
      if (length(levels(d)) < max.fact.no - 1) {
        return(levels(d))
      }
      c(names(sort(table(d), decreasing = TRUE))[seq_len(max.fact.no - 1)],
        rep("collapsed", length(levels(d)) - max.fact.no + 1))
    }, simplify = FALSE)
  },
  cpo.retrafo = {
    for (n in names(data)) {
      levels(data[[n]]) = control[[n]]
    }
    data
  })

# Make the preprocessing + learner pipeline
make_preproc_pipeline = function(algo) {
  algo = sanitize_algo(algo)
  pipe = cpoFixFactors() %>>%
    cpoCbind(
        cpoImputeConstant("__MISSING__", affect.type = c("factor", "ordered")) %>>%
        cpoMultiplex(id = "num.impute",
          list(
              cpoImputeMean(affect.type = "numeric"),
              cpoImputeMedian(affect.type = "numeric"),
              cpoImputeHist(use.mids = FALSE, affect.type = "numeric")),
          selected.cpo = "impute.hist"),
        MISSING = cpoSelect(type = "numeric") %>>% cpoMissingIndicators()) %>>%
    cpoMaxFact(32) %>>%
    cpoDropConstants(abs.tol = 0)

    if (algo %in% c("classif.knn", "classif.xgboost"))
      pipe = pipe %>>% cpoDummyEncode(reference.cat = TRUE, infixdot = TRUE)

    if (algo != "classif.knn") {
      mlr_algo_name = algo
    } else {
      mlr_algo_name = "classif.RcppHNSW"
    }

    lrn = makeLearner(mlr_algo_name, predict.type = "prob")
    pipe %>>% lrn
}
