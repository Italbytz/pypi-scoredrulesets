#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(glmnet)
})

usage <- function() {
  cat(
    "Usage:\n",
    "  Rscript scripts/r/select_optimal_k_nested_cv.R \\\n",
    "    --data <input.csv> --outdir <output-dir> [options]\n\n",
    "Options:\n",
    "  --seed <int>                 Default: 20260425\n",
    "  --outer-folds <int>          Default: 5\n",
    "  --inner-folds <int>          Default: 5\n",
    "  --k-grid <csv>               Default: 16,32,48,63,96,128\n",
    "  --stability-runs <int>       Default: 100\n",
    "  --stability-threshold <num>  Default: 0.60\n",
    "  --maf-min <num>              Default: 0.01\n",
    "  --missing-max <num>          Default: 0.02\n",
    sep = ""
  )
}

parse_args <- function(args) {
  if (length(args) %% 2 != 0) stop("Arguments must be provided as --key value pairs.")
  keys <- args[seq(1, length(args), by = 2)]
  vals <- args[seq(2, length(args), by = 2)]
  out <- as.list(vals)
  names(out) <- keys
  out
}

get_or_default <- function(lst, key, default) {
  if (!is.null(lst[[key]])) lst[[key]] else default
}

to_int <- function(x, name) {
  v <- suppressWarnings(as.integer(x))
  if (is.na(v)) stop(sprintf("Invalid integer for %s: %s", name, x))
  v
}

to_num <- function(x, name) {
  v <- suppressWarnings(as.numeric(x))
  if (is.na(v)) stop(sprintf("Invalid numeric for %s: %s", name, x))
  v
}

balanced_accuracy <- function(y_true, y_pred) {
  p <- sum(y_true == 1)
  n <- sum(y_true == 0)
  if (p == 0 || n == 0) return(NA_real_)
  tpr <- sum(y_pred == 1 & y_true == 1) / p
  tnr <- sum(y_pred == 0 & y_true == 0) / n
  0.5 * (tpr + tnr)
}

stratified_folds <- function(y, k, seed_offset = 0L) {
  set.seed(seed_offset)
  idx0 <- sample(which(y == 0))
  idx1 <- sample(which(y == 1))
  folds <- vector("list", k)
  for (i in seq_len(k)) {
    part0 <- idx0[seq(i, length(idx0), by = k)]
    part1 <- idx1[seq(i, length(idx1), by = k)]
    folds[[i]] <- c(part0, part1)
  }
  folds
}

compute_maf <- function(g) {
  g2 <- g[!is.na(g)]
  if (length(g2) == 0) return(0)
  p <- mean(g2) / 2
  min(p, 1 - p)
}

train_qc_mask <- function(X_train, maf_min, missing_max) {
  miss <- colMeans(is.na(X_train))
  maf <- apply(X_train, 2, compute_maf)
  keep <- (miss <= missing_max) & (maf >= maf_min)
  keep
}

mode_impute <- function(v) {
  w <- v[!is.na(v)]
  if (length(w) == 0) return(0)
  tab <- table(w)
  as.numeric(names(tab)[which.max(tab)])
}

impute_train_test <- function(X_train, X_test) {
  modes <- apply(X_train, 2, mode_impute)
  for (j in seq_len(ncol(X_train))) {
    X_train[is.na(X_train[, j]), j] <- modes[j]
    X_test[is.na(X_test[, j]), j] <- modes[j]
  }
  list(train = X_train, test = X_test)
}

rank_snps_chisq <- function(X, y) {
  pvals <- rep(1, ncol(X))
  for (j in seq_len(ncol(X))) {
    xj <- X[, j]
    ok <- !is.na(xj)
    if (length(unique(xj[ok])) < 2) {
      pvals[j] <- 1
      next
    }
    tab <- table(factor(xj[ok], levels = c(0, 1, 2)), factor(y[ok], levels = c(0, 1)))
    pv <- tryCatch(suppressWarnings(chisq.test(tab, correct = FALSE)$p.value), error = function(e) 1)
    pvals[j] <- ifelse(is.na(pv), 1, pv)
  }
  order(pvals, decreasing = FALSE)
}

fit_predict_ridge <- function(X_train, y_train, X_test, seed_offset = 0L) {
  set.seed(seed_offset)
  nfolds <- min(5L, sum(y_train == 0), sum(y_train == 1))
  if (nfolds < 2) return(rep(round(mean(y_train)), nrow(X_test)))

  cvfit <- cv.glmnet(
    x = X_train,
    y = y_train,
    family = "binomial",
    alpha = 0,
    nfolds = nfolds,
    type.measure = "class",
    standardize = TRUE
  )

  probs <- as.numeric(predict(cvfit, newx = X_test, s = "lambda.1se", type = "response"))
  as.integer(probs >= 0.5)
}

eval_k_inner <- function(X, y, inner_folds, k_grid, maf_min, missing_max, seed_base) {
  scores <- matrix(NA_real_, nrow = length(k_grid), ncol = length(inner_folds))
  rownames(scores) <- as.character(k_grid)

  for (f in seq_along(inner_folds)) {
    val_idx <- inner_folds[[f]]
    tr_idx <- setdiff(seq_len(nrow(X)), val_idx)

    X_tr <- X[tr_idx, , drop = FALSE]
    y_tr <- y[tr_idx]
    X_val <- X[val_idx, , drop = FALSE]
    y_val <- y[val_idx]

    keep <- train_qc_mask(X_tr, maf_min = maf_min, missing_max = missing_max)
    if (!any(keep)) next

    X_tr <- X_tr[, keep, drop = FALSE]
    X_val <- X_val[, keep, drop = FALSE]

    ranked <- rank_snps_chisq(X_tr, y_tr)

    for (i in seq_along(k_grid)) {
      k <- min(k_grid[i], length(ranked))
      if (k < 2) next
      sel <- ranked[seq_len(k)]
      xx <- impute_train_test(X_tr[, sel, drop = FALSE], X_val[, sel, drop = FALSE])
      pred <- fit_predict_ridge(xx$train, y_tr, xx$test, seed_offset = seed_base + f * 100 + i)
      scores[i, f] <- balanced_accuracy(y_val, pred)
    }
  }

  rowMeans(scores, na.rm = TRUE)
}

pick_k <- function(k_grid, mean_scores) {
  if (all(is.na(mean_scores))) stop("All inner scores are NA; cannot choose k.")
  best <- max(mean_scores, na.rm = TRUE)
  candidates <- k_grid[which(abs(mean_scores - best) < 1e-12)]
  min(candidates)
}

stratified_bootstrap <- function(y, seed_offset = 0L) {
  set.seed(seed_offset)
  idx0 <- which(y == 0)
  idx1 <- which(y == 1)
  c(sample(idx0, length(idx0), replace = TRUE), sample(idx1, length(idx1), replace = TRUE))
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 0 || any(args %in% c("--help", "-h"))) {
    usage()
    quit(status = 0)
  }

  parsed <- parse_args(args)

  data_path <- parsed[["--data"]]
  outdir <- parsed[["--outdir"]]
  if (is.null(data_path) || is.null(outdir)) {
    usage()
    stop("--data and --outdir are required.")
  }

  seed <- to_int(get_or_default(parsed, "--seed", "20260425"), "--seed")
  outer_folds_n <- to_int(get_or_default(parsed, "--outer-folds", "5"), "--outer-folds")
  inner_folds_n <- to_int(get_or_default(parsed, "--inner-folds", "5"), "--inner-folds")
  k_grid <- as.integer(strsplit(get_or_default(parsed, "--k-grid", "16,32,48,63,96,128"), ",")[[1]])
  k_grid <- sort(unique(k_grid[!is.na(k_grid) & k_grid > 1]))
  if (length(k_grid) == 0) stop("k-grid must contain integers > 1")

  stability_runs <- to_int(get_or_default(parsed, "--stability-runs", "100"), "--stability-runs")
  stability_threshold <- to_num(get_or_default(parsed, "--stability-threshold", "0.60"), "--stability-threshold")
  maf_min <- to_num(get_or_default(parsed, "--maf-min", "0.01"), "--maf-min")
  missing_max <- to_num(get_or_default(parsed, "--missing-max", "0.02"), "--missing-max")

  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

  raw <- read.csv2(data_path, stringsAsFactors = FALSE)
  if (ncol(raw) < 3) stop("Need at least one label column and two SNP columns.")

  y <- as.integer(raw[, 1])
  if (!all(y %in% c(0L, 1L))) stop("Label column must contain only 0/1 values.")

  X <- as.matrix(raw[, -1, drop = FALSE])
  X <- matrix(as.integer(X) - 1L, nrow = nrow(X), dimnames = list(NULL, colnames(raw)[-1]))

  set.seed(seed)
  outer_folds <- stratified_folds(y, outer_folds_n, seed_offset = seed)

  outer_rows <- list()
  selected_snps_by_fold <- list()

  for (o in seq_along(outer_folds)) {
    test_idx <- outer_folds[[o]]
    train_idx <- setdiff(seq_len(nrow(X)), test_idx)

    X_outer_tr <- X[train_idx, , drop = FALSE]
    y_outer_tr <- y[train_idx]
    X_outer_te <- X[test_idx, , drop = FALSE]
    y_outer_te <- y[test_idx]

    inner_folds <- stratified_folds(y_outer_tr, inner_folds_n, seed_offset = seed + o * 1000L)
    inner_scores <- eval_k_inner(
      X = X_outer_tr,
      y = y_outer_tr,
      inner_folds = inner_folds,
      k_grid = k_grid,
      maf_min = maf_min,
      missing_max = missing_max,
      seed_base = seed + o * 10000L
    )

    k_opt <- pick_k(k_grid, inner_scores)

    keep <- train_qc_mask(X_outer_tr, maf_min = maf_min, missing_max = missing_max)
    if (!any(keep)) stop(sprintf("No SNPs left after QC in outer fold %d", o))

    X_tr <- X_outer_tr[, keep, drop = FALSE]
    X_te <- X_outer_te[, keep, drop = FALSE]
    ranked <- rank_snps_chisq(X_tr, y_outer_tr)
    k_use <- min(k_opt, length(ranked))
    sel <- ranked[seq_len(k_use)]

    xx <- impute_train_test(X_tr[, sel, drop = FALSE], X_te[, sel, drop = FALSE])
    pred <- fit_predict_ridge(xx$train, y_outer_tr, xx$test, seed_offset = seed + o * 20000L)
    ba <- balanced_accuracy(y_outer_te, pred)

    snp_names <- colnames(X_tr)[sel]
    selected_snps_by_fold[[o]] <- snp_names

    row <- data.frame(
      outer_fold = o,
      n_train = length(train_idx),
      n_test = length(test_idx),
      k_opt = k_opt,
      k_used = k_use,
      balanced_accuracy = ba,
      stringsAsFactors = FALSE
    )
    outer_rows[[o]] <- row

    inner_df <- data.frame(
      outer_fold = o,
      k = k_grid,
      mean_inner_balanced_accuracy = inner_scores,
      stringsAsFactors = FALSE
    )
    write.csv(inner_df, file.path(outdir, sprintf("inner_scores_fold_%02d.csv", o)), row.names = FALSE)

    writeLines(snp_names, con = file.path(outdir, sprintf("selected_snps_fold_%02d.txt", o)))
  }

  outer_df <- do.call(rbind, outer_rows)
  write.csv(outer_df, file.path(outdir, "outer_results.csv"), row.names = FALSE)

  k_final <- as.integer(names(sort(table(outer_df$k_opt), decreasing = TRUE))[1])

  freq <- table(unlist(selected_snps_by_fold))
  freq_df <- data.frame(
    snp = names(freq),
    fold_count = as.integer(freq),
    fold_frequency = as.numeric(freq) / length(selected_snps_by_fold),
    stringsAsFactors = FALSE
  )
  freq_df <- freq_df[order(-freq_df$fold_frequency, freq_df$snp), ]
  write.csv(freq_df, file.path(outdir, "outer_fold_snp_frequencies.csv"), row.names = FALSE)

  # Stability selection using fixed final k.
  stable_counter <- integer(0)
  names(stable_counter) <- character(0)

  for (b in seq_len(stability_runs)) {
    idx <- stratified_bootstrap(y, seed_offset = seed + b * 30000L)
    Xb <- X[idx, , drop = FALSE]
    yb <- y[idx]

    keep <- train_qc_mask(Xb, maf_min = maf_min, missing_max = missing_max)
    if (!any(keep)) next

    Xb <- Xb[, keep, drop = FALSE]
    ranked <- rank_snps_chisq(Xb, yb)
    k_use <- min(k_final, length(ranked))
    if (k_use < 2) next

    snps <- colnames(Xb)[ranked[seq_len(k_use)]]
    for (s in snps) {
      if (is.na(stable_counter[s])) stable_counter[s] <- 0L
      stable_counter[s] <- stable_counter[s] + 1L
    }
  }

  if (length(stable_counter) > 0) {
    stable_df <- data.frame(
      snp = names(stable_counter),
      bootstrap_count = as.integer(stable_counter),
      bootstrap_frequency = as.numeric(stable_counter) / stability_runs,
      stringsAsFactors = FALSE
    )
    stable_df <- stable_df[order(-stable_df$bootstrap_frequency, stable_df$snp), ]
    write.csv(stable_df, file.path(outdir, "stability_selection.csv"), row.names = FALSE)

    stable_cut <- stable_df[stable_df$bootstrap_frequency >= stability_threshold, , drop = FALSE]
    write.csv(stable_cut, file.path(outdir, "stable_snps_thresholded.csv"), row.names = FALSE)
  }

  summary_lines <- c(
    sprintf("data=%s", normalizePath(data_path, mustWork = TRUE)),
    sprintf("seed=%d", seed),
    sprintf("outer_folds=%d", outer_folds_n),
    sprintf("inner_folds=%d", inner_folds_n),
    sprintf("k_grid=%s", paste(k_grid, collapse = ",")),
    sprintf("maf_min=%.6f", maf_min),
    sprintf("missing_max=%.6f", missing_max),
    sprintf("stability_runs=%d", stability_runs),
    sprintf("stability_threshold=%.4f", stability_threshold),
    sprintf("k_final=%d", k_final),
    sprintf("outer_balanced_accuracy_mean=%.6f", mean(outer_df$balanced_accuracy, na.rm = TRUE)),
    sprintf("outer_balanced_accuracy_sd=%.6f", sd(outer_df$balanced_accuracy, na.rm = TRUE)),
    sprintf("r_version=%s", R.version.string),
    sprintf("glmnet_version=%s", as.character(packageVersion("glmnet")))
  )
  writeLines(summary_lines, con = file.path(outdir, "run_manifest.txt"))

  cat("Nested CV k-selection finished\n")
  cat("Output directory:", outdir, "\n")
  cat("Final k:", k_final, "\n")
  cat("Outer mean balanced accuracy:", sprintf("%.4f", mean(outer_df$balanced_accuracy, na.rm = TRUE)), "\n")
}

main()
