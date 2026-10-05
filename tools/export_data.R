suppressPackageStartupMessages(library(arrow))

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
  stop("Usage: Rscript tools/export_data.R <path to the eventclock R repo>")
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

out_dir <- file.path(py_root, "src", "eventclock", "data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

rda_files <- list.files(file.path(repo_root, "data"), pattern = "\\.rda$", full.names = TRUE)

for (f in rda_files) {
  e <- new.env()
  objs <- load(f, envir = e)
  for (nm in objs) {
    d <- as.data.frame(get(nm, envir = e))
    for (col in names(d)) {
      if (inherits(d[[col]], "POSIXct")) {
        attr(d[[col]], "tzone") <- "UTC"
      }
    }
    tbl <- arrow::arrow_table(d)
    tbl$metadata$r <- NULL

    for (col in names(d)) {
      ftype <- tbl$schema[[col]]$type
      if (inherits(d[[col]], "Date") && ftype$ToString() != "date32[day]") {
        stop(nm, "$", col, ": expected date32, got ", ftype$ToString())
      }
      if (inherits(d[[col]], "POSIXct") && !grepl("^timestamp\\[.*, tz=UTC\\]$", ftype$ToString())) {
        stop(nm, "$", col, ": expected UTC timestamp, got ", ftype$ToString())
      }
    }

    path <- file.path(out_dir, paste0(nm, ".parquet"))
    arrow::write_parquet(tbl, path)

    back <- as.data.frame(arrow::read_parquet(path))
    ok <- isTRUE(all.equal(back, d, check.attributes = FALSE, tolerance = 0))
    if (!ok || !identical(names(back), names(d)) || nrow(back) != nrow(d)) {
      stop(nm, ": parquet round trip differs from the .rda data.")
    }
    cat(sprintf("%-16s %6d rows  %s\n", nm, nrow(d), path))
  }
}