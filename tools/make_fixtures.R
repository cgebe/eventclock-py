expected_sha <- "30781dd48a691dbdb253b6c10463e3b2d1731aed"

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
tools_dir <- if (length(script_arg) == 1) {
  dirname(normalizePath(sub("^--file=", "", script_arg)))
} else {
  normalizePath("tools")
}
py_root <- normalizePath(file.path(tools_dir, ".."))

r_repo_arg <- commandArgs(trailingOnly = TRUE)
if (length(r_repo_arg) != 1) {
  stop("Usage: Rscript tools/make_fixtures.R <path to the eventclock R repo>")
}
repo_root <- normalizePath(r_repo_arg, mustWork = TRUE)

sha <- system2("git", c("-C", repo_root, "rev-parse", "HEAD"), stdout = TRUE)
if (!identical(sha, expected_sha)) {
  stop("Repo is at ", sha, ", expected ", expected_sha, ".")
}
dirty <- system2("git", c("-C", repo_root, "status", "--porcelain", "--", "R", "data", "DESCRIPTION"),
                 stdout = TRUE)
if (length(dirty) > 0) {
  stop("R/, data/ or DESCRIPTION has local changes:\n", paste(dirty, collapse = "\n"))
}

if (requireNamespace("pkgload", quietly = TRUE)) {
  pkgload::load_all(repo_root, quiet = TRUE, export_all = FALSE)
} else {
  stop("Install pkgload, so that the fixtures use the R code at ", expected_sha, ".")
}
suppressPackageStartupMessages(library(jsonlite))

`%||%` <- function(a, b) if (is.null(a)) b else a

options(cli.width = 10000, cli.num_colors = 1)

out_dir <- file.path(py_root, "tests", "fixtures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
unlink(list.files(out_dir, pattern = "\\.json$", full.names = TRUE))

ds <- function(nm) getExportedValue("eventclock", nm)

fmt_date <- function(x) format(x, "%Y-%m-%d")

fmt_instant <- function(x) {
  secs <- as.numeric(x)
  whole <- all(is.na(secs) | secs == floor(secs))
  format(x, if (whole) "%Y-%m-%dT%H:%M:%SZ" else "%Y-%m-%dT%H:%M:%OS6Z", tz = "UTC")
}

col_type <- function(x) {
  if (inherits(x, "Date")) return("date")
  if (inherits(x, "POSIXct")) return("instant")
  if (is.logical(x)) return("logical")
  if (is.integer(x)) return("integer")
  if (is.double(x)) return("double")
  if (is.character(x)) return("character")
  stop("Unsupported column type: ", paste(class(x), collapse = "/"))
}

enc_col <- function(x) {
  v <- switch(col_type(x),
    date = fmt_date(x),
    instant = fmt_instant(x),
    as.vector(unclass(x))
  )
  if (is.double(v) && any(is.infinite(v))) stop("Infinite value cannot go to JSON.")
  I(unname(v))
}

enc_value <- function(v) {
  if (is.null(v)) return(NULL)
  nms <- names(v)
  wrap <- function(vals) {
    if (!is.null(nms)) as.list(stats::setNames(vals, nms)) else unname(vals)
  }
  if (inherits(v, "Date")) return(list(date = wrap(fmt_date(v))))
  if (inherits(v, "POSIXct")) {
    return(list(instant = wrap(fmt_instant(v)), tz = attr(v, "tzone") %||% ""))
  }
  if (is.function(v)) stop("Function arguments cannot go to JSON.")
  wrap(as.vector(unclass(v)))
}

enc_df <- function(d) {
  base <- as.data.frame(d)
  extra <- setdiff(names(attributes(d)), c("names", "row.names", "class"))
  types <- lapply(base, col_type)
  tz <- lapply(base[unlist(types) == "instant"], function(x) attr(x, "tzone") %||% "")
  list(
    class = class(d),
    nrow = nrow(base),
    types = types,
    tz = if (length(tz)) tz else setNames(list(), character()),
    columns = lapply(base, enc_col),
    attrs = if (length(extra)) lapply(attributes(d)[extra], enc_value) else setNames(list(), character())
  )
}

enc_output <- function(res) {
  if (inherits(res, "event_prices")) {
    out <- list(
      event_prices = enc_df(res),
      time_kind = if (inherits(res$time, "Date")) "date" else "instant",
      summary = enc_df(summary(res))
    )
    return(out)
  }
  if (inherits(res, "ec_validation")) {
    return(list(table = enc_df(res), printed = I(utils::capture.output(print(res)))))
  }
  if (is.data.frame(res)) return(list(table = enc_df(res)))
  stop("Unsupported output class: ", paste(class(res), collapse = "/"))
}

clean_msg <- function(m) trimws(cli::ansi_strip(conditionMessage(m)))

run <- function(name, fn, input, args = list(), input_ep_args = NULL) {
  x <- switch(input$kind,
    dataset = ds(input$name),
    event_prices = ep[[input$name]],
    rows = do.call(as_event_prices, c(list(input$data), input_ep_args %||% list()))
  )
  warns <- character()
  msgs <- character()
  res <- withCallingHandlers(
    do.call(fn, c(list(x), args)),
    warning = function(w) {
      warns <<- c(warns, clean_msg(w))
      invokeRestart("muffleWarning")
    },
    message = function(m) {
      msgs <<- c(msgs, clean_msg(m))
      invokeRestart("muffleMessage")
    }
  )
  inp <- switch(input$kind,
    dataset = list(kind = "dataset", name = input$name),
    event_prices = list(kind = "event_prices", name = input$name),
    rows = list(kind = "rows", data = enc_df(input$data),
                as_event_prices_args = lapply(input_ep_args %||% list(), enc_value))
  )
  fixture <- list(
    name = name,
    source_sha = expected_sha,
    r_version = R.version.string,
    `function` = fn,
    input = inp,
    args = lapply(args, enc_value),
    warnings = I(warns),
    messages = I(msgs),
    output = enc_output(res)
  )
  path <- file.path(out_dir, paste0(name, ".json"))
  jsonlite::write_json(fixture, path,
    digits = I(17), na = "null", null = "null", auto_unbox = TRUE, pretty = FALSE
  )
  cat(sprintf("%-26s %s\n", name, basename(path)))
  invisible(res)
}

brexit_horizons <- c(
  `1W` = as.Date("2016-05-31"), `2W` = as.Date("2016-06-07"),
  `1M` = as.Date("2016-06-23")
)
us_horizons <- c(
  `1W` = as.Date("2016-10-17"), `2W` = as.Date("2016-10-24"),
  `1M` = as.Date("2016-11-08")
)

ep_args <- list(
  ep_brexit = list(
    dataset = "brexit2016",
    args = list(time = "date", price = "q_leave",
                market_id = "Brexit: Leave", event_date = as.Date("2016-06-23"))
  ),
  ep_us = list(
    dataset = "us2016",
    args = list(time = "date", price = "trump",
                market_id = "US 2016: Trump", event_date = as.Date("2016-11-08"))
  ),
  ep_pm2024 = list(dataset = "polymarket2024", args = list())
)

ep <- list()
for (nm in names(ep_args)) {
  ep[[nm]] <- run(nm, "as_event_prices",
    list(kind = "dataset", name = ep_args[[nm]]$dataset), ep_args[[nm]]$args)
}

ny_rows <- tibble::tibble(
  time = as.POSIXct("2020-01-01 12:00", tz = "America/New_York") + 86400 * (0:4),
  q = c(0.40, 0.45, 0.50, 0.55, 0.60)
)
gap_rows <- tibble::tibble(
  time = as.Date("2020-01-01") + 0:5,
  q = c(0.50, 0.52, NA, 0.55, 0.57, 0.60)
)
validate_rows <- tibble::tibble(
  time = as.Date("2020-01-01") + c(0:5, 8:12),
  q = c(0.40, 0.40, 0.40, 0.41, NA, 0.415, 0.42, 0.42, 0.005, 0.42, 0.43)
)

E <- function(nm) list(kind = "event_prices", name = nm)
Rows <- function(d) list(kind = "rows", data = d)

run("clock_brexit", "event_clock", E("ep_brexit"),
    list(from = as.Date("2016-05-24"), to = brexit_horizons))
run("clock_us", "event_clock", E("ep_us"),
    list(from = as.Date("2016-10-10"), to = us_horizons))
run("clock_brexit_se", "event_clock", E("ep_brexit"),
    list(from = as.Date("2016-05-24"), to = brexit_horizons, methods = "rv", se = TRUE))
for (k in c(2L, 3L, 5L)) {
  run(paste0("clock_brexit_k", k), "event_clock", E("ep_brexit"), list(sample_every = k))
}
run("clock_pm2024", "event_clock", E("ep_pm2024"))
run("clock_pm2024_datebounds", "event_clock", E("ep_pm2024"),
    list(from = as.Date("2024-10-01"), to = as.Date("2024-11-05")))
run("clock_ny_datebounds", "event_clock", Rows(ny_rows),
    list(from = as.Date("2020-01-02"), to = as.Date("2020-01-04")))
run("clock_gaps", "event_clock", Rows(gap_rows))

run("path_brexit", "event_clock_path", E("ep_brexit"))
run("path_pm2024", "event_clock_path", E("ep_pm2024"))
run("path_ny_datebounds", "event_clock_path", Rows(ny_rows),
    list(from = as.Date("2020-01-02"), to = as.Date("2020-01-04")))

for (tr in c(20L, 40L, 60L)) {
  run(paste0("forecast_brexit_t", tr), "event_clock_forecast", E("ep_brexit"),
      list(at = as.Date("2016-05-24"), horizon = c(`1W` = 7, `2W` = 14), trailing = tr))
  run(paste0("forecast_us_t", tr), "event_clock_forecast", E("ep_us"),
      list(at = as.Date("2016-10-10"), horizon = c(`1W` = 7, `2W` = 14), trailing = tr))
}

run("sig_brexit", "ec_signature", E("ep_brexit"), list(max_every = 10L))
run("sig_pm2024", "ec_signature", E("ep_pm2024"), list(max_every = 24L))

run("validate_brexit", "ec_validate", E("ep_brexit"))
run("validate_us", "ec_validate", E("ep_us"))
run("validate_pm2024", "ec_validate", E("ep_pm2024"))
run("validate_constructed", "ec_validate", Rows(validate_rows))

run("daily_pm2024", "pm_daily", E("ep_pm2024"))
run("daily_pm2024_utc", "pm_daily", E("ep_pm2024"), list(tz = "UTC", snapshot_hour = 15.5))

cat(length(list.files(out_dir, pattern = "\\.json$")), "fixtures in", out_dir, "\n")