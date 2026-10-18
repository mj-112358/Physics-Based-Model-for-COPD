if (!requireNamespace("rspiro", quietly = TRUE)) {
  install.packages("rspiro", repos = "https://cloud.r-project.org")
}

library(rspiro)

root <- getwd()
results_dir <- file.path(root, "results")
input_path <- file.path(results_dir, "_gli_input.csv")
output_path <- file.path(results_dir, "gli_reference.csv")
notes_path <- file.path(results_dir, "gli_mapping_notes.md")
meta_path <- file.path(results_dir, "gli_reference_meta.csv")

if (!file.exists(input_path)) {
  stop("Missing GLI input CSV: ", input_path)
}

validate_rspiro_examples <- function() {
  # Values copied from rspiro 0.5 package examples for Caucasian women,
  # age 20:70, height 1.70 m, sex=2. These validate that the package API,
  # units, and output scale match documented GLI-2012 examples.
  ages <- c(20, 30, 70)
  expected_fev1 <- c(3.661443, 3.532514, 2.429129)
  expected_fvc <- c(4.197767, 4.203597, 3.153356)
  pred <- pred_GLI(ages, 1.7, 2, 1, param = c("FEV1", "FVC"))
  fev1_ok <- abs(pred$pred.FEV1 - expected_fev1) / expected_fev1 <= 0.02
  fvc_ok <- abs(pred$pred.FVC - expected_fvc) / expected_fvc <= 0.02
  if (!all(fev1_ok) || !all(fvc_ok)) {
    stop("rspiro validation failed against documented pred_GLI examples.")
  }
}

map_ethnicity <- function(race_ethnicity) {
  # NHANES race/ethnicity is not isomorphic to GLI-2012 ethnicity.
  # NHANES code 3, Non-Hispanic White -> GLI 1, Caucasian.
  # NHANES code 4, Non-Hispanic Black -> GLI 2, African-American.
  # NHANES code 1, Mexican-American -> GLI 5, Other/mixed fallback.
  # NHANES code 2, Other Hispanic -> GLI 5, Other/mixed fallback.
  # NHANES code 5, Other Race including Multi-Racial in RIDRETH1 -> GLI 5 fallback.
  # NHANES code 6, Non-Hispanic Asian in RIDRETH3 has no NE/SE Asian split -> GLI 5 fallback.
  # NHANES code 7, Other Race including Multi-Racial in RIDRETH3 -> GLI 5 fallback.
  out <- rep(5L, length(race_ethnicity))
  out[race_ethnicity == 3] <- 1L
  out[race_ethnicity == 4] <- 2L
  out
}

validate_rspiro_examples()

input <- read.csv(input_path, stringsAsFactors = FALSE)
input$height_m <- input$height_cm / 100
input$gli_ethnicity_mapped <- map_ethnicity(input$race_ethnicity)
input$gli_mapping_fallback <- !(input$race_ethnicity %in% c(3, 4))

pred <- pred_GLI(
  age = input$age_years,
  height = input$height_m,
  gender = input$sex,
  ethnicity = input$gli_ethnicity_mapped,
  param = c("FEV1", "FVC", "FEV1FVC")
)
lln_ratio <- LLN_GLI(
  age = input$age_years,
  height = input$height_m,
  gender = input$sex,
  ethnicity = input$gli_ethnicity_mapped,
  param = "FEV1FVC"
)

out <- data.frame(
  SEQN = input$SEQN,
  gli_pred_fev1 = pred$pred.FEV1,
  gli_pred_fvc = pred$pred.FVC,
  gli_pred_ratio = pred$pred.FEV1FVC,
  gli_lln_ratio = lln_ratio,
  gli_ethnicity_mapped = input$gli_ethnicity_mapped,
  gli_mapping_fallback = input$gli_mapping_fallback
)

finite <- is.finite(out$gli_pred_fev1) &
  is.finite(out$gli_pred_fvc) &
  is.finite(out$gli_pred_ratio) &
  is.finite(out$gli_lln_ratio)
sane_fev1 <- out$gli_pred_fev1 >= 0.5 & out$gli_pred_fev1 <= 6.0
sanity_drop_count <- sum(finite & !sane_fev1)
out <- out[finite & sane_fev1, ]

write.csv(out, output_path, row.names = FALSE, na = "")

fallback_count <- sum(input$gli_mapping_fallback, na.rm = TRUE)
fallback_output_count <- sum(out$gli_mapping_fallback, na.rm = TRUE)
nonfinite_count <- nrow(input) - nrow(out)
nonfinite_rspiro_count <- sum(!finite)

notes <- c(
  "# GLI Mapping Notes",
  "",
  paste0("- rspiro version: ", as.character(packageVersion("rspiro"))),
  "- GLI functions called: `pred_GLI(age, height, gender, ethnicity, param=c(\"FEV1\",\"FVC\",\"FEV1FVC\"))`; `LLN_GLI(..., param=\"FEV1FVC\")`.",
  "- rspiro documentation states height is in meters; `height_cm` was divided by 100.",
  "- Validation: rspiro documented `pred_GLI` examples for Caucasian women aged 20, 30, and 70 at height 1.70 m matched within 2%.",
  "",
  "## NHANES to GLI Ethnicity Mapping",
  "",
  "| NHANES code | Interpretation | GLI code | Fallback |",
  "|---:|---|---:|---|",
  "| 1 | Mexican-American | 5 Other/mixed | yes |",
  "| 2 | Other Hispanic | 5 Other/mixed | yes |",
  "| 3 | Non-Hispanic White | 1 Caucasian | no |",
  "| 4 | Non-Hispanic Black | 2 African-American | no |",
  "| 5 | Other Race including Multi-Racial (RIDRETH1) | 5 Other/mixed | yes |",
  "| 6 | Non-Hispanic Asian (RIDRETH3; no NE/SE split) | 5 Other/mixed | yes |",
  "| 7 | Other Race including Multi-Racial (RIDRETH3) | 5 Other/mixed | yes |",
  "",
  paste0("- Input rows using fallback GLI Other/mixed mapping: ", fallback_count),
  paste0("- Output rows using fallback GLI Other/mixed mapping: ", fallback_output_count),
  paste0("- Input rows dropped because rspiro returned non-finite GLI values: ", nonfinite_rspiro_count),
  paste0("- Finite GLI rows dropped by predicted FEV1 sanity range 0.5-6 L: ", sanity_drop_count)
)
writeLines(notes, notes_path)

meta <- data.frame(
  key = c(
    "rspiro_version",
    "gli_functions",
    "input_rows",
    "output_rows",
    "fallback_input_rows",
    "fallback_output_rows",
    "dropped_nonfinite_gli_rows",
    "dropped_pred_fev1_sanity_rows"
  ),
  value = c(
    as.character(packageVersion("rspiro")),
    "pred_GLI; LLN_GLI",
    as.character(nrow(input)),
    as.character(nrow(out)),
    as.character(fallback_count),
    as.character(fallback_output_count),
    as.character(nonfinite_rspiro_count),
    as.character(sanity_drop_count)
  )
)
write.csv(meta, meta_path, row.names = FALSE)

cat("rspiro version:", as.character(packageVersion("rspiro")), "\n")
cat("GLI functions: pred_GLI; LLN_GLI\n")
cat("GLI reference rows written:", nrow(out), "\n")
