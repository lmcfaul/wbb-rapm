#!/usr/bin/env Rscript
# Stage 1: pull wehoop data for one or more seasons and cache to data/raw/.
# Usage: Rscript R/fetch_data.R --season 2026 [--season 2025 ...] [--refresh]
# wehoop season convention: 2026 == the 2025-26 season.
# Writes parquet if the arrow package is available, otherwise csv.gz.

suppressPackageStartupMessages({
  library(wehoop)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
seasons <- as.integer(args[which(args == "--season") + 1])
refresh <- "--refresh" %in% args
if (length(seasons) == 0 || any(is.na(seasons))) {
  stop("Usage: Rscript R/fetch_data.R --season YEAR [--season YEAR ...] [--refresh]")
}

has_arrow <- requireNamespace("arrow", quietly = TRUE)
ext <- if (has_arrow) "parquet" else "csv.gz"
raw_dir <- file.path("data", "raw")
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)

write_cache <- function(df, name, season) {
  path <- file.path(raw_dir, sprintf("%s_%d.%s", name, season, ext))
  if (file.exists(path) && !refresh) {
    message(sprintf("[skip] %s exists", path))
    return(invisible(NULL))
  }
  # Flatten list-columns that neither parquet-via-csv nor csv can hold.
  df <- df %>% mutate(across(where(is.list), ~ sapply(., paste, collapse = ";")))
  if (has_arrow) {
    arrow::write_parquet(df, path)
  } else {
    readr::write_csv(df, path)
  }
  message(sprintf("[ok] %s (%d rows)", path, nrow(df)))
}

for (season in seasons) {
  message(sprintf("=== season %d ===", season))
  write_cache(load_wbb_pbp(seasons = season), "pbp", season)
  write_cache(load_wbb_player_box(seasons = season), "player_box", season)
  write_cache(load_wbb_team_box(seasons = season), "team_box", season)
  teams <- espn_wbb_teams(year = season)
  write_cache(teams, "teams", season)
}
message("done")
