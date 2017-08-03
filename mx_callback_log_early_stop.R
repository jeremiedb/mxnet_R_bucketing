mx.callback.log.early.stop <- function(period, early_stop_rounds, maximize, logger=NULL) {
  function(iteration, nbatch, env, verbose) {
    if (nbatch %% period == 0 && !is.null(env$metric)) {
      result <- env$metric$get(env$train.metric)
      if (nbatch != 0)
        if(verbose) cat(paste0("Batch [", nbatch, "] Train-", result$name, "=", result$value, "\n"))
      if (!is.null(logger)) {
        if (class(logger) != "mx.metric.logger") {
          stop("Invalid mx.metric.logger.")
        }
        logger$train <- c(logger$train, result$value)
        if (!is.null(env$eval.metric)) {
          result <- env$metric$get(env$eval.metric)
          if (nbatch != 0)
            cat(paste0("Batch [", nbatch, "] Validation-", result$name, "=", result$value, "\n"))
          logger$eval <- c(logger$eval, result$value)
          
          if (maximize) target_eval_id <- which.max(logger$eval) else 
            target_eval_id <- which.min(logger$eval)
          
          if (length(logger$eval) - target_eval_id == early_stop_rounds) return(FALSE)
        }
      }
    }
    return(TRUE)
  }
}
