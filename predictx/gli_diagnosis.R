if (!requireNamespace("rspiro", quietly = TRUE)) {
  stop("rspiro is not installed; cannot diagnose GLI LLN output.")
}

library(rspiro)

people <- data.frame(
  person = c("A", "B", "C"),
  age_years = c(65, 25, 65),
  sex = c(1, 1, 2),
  height_cm = c(175, 175, 160),
  ethnicity = c(1, 1, 1),
  description = c(
    "65yo male, 175cm, Caucasian",
    "25yo male, 175cm, Caucasian",
    "65yo female, 160cm, Caucasian"
  )
)

people$height_m <- people$height_cm / 100

pred <- pred_GLI(
  age = people$age_years,
  height = people$height_m,
  gender = people$sex,
  ethnicity = people$ethnicity,
  param = c("FEV1", "FVC", "FEV1FVC")
)

lln_fev1 <- LLN_GLI(
  age = people$age_years,
  height = people$height_m,
  gender = people$sex,
  ethnicity = people$ethnicity,
  param = "FEV1"
)
lln_fvc <- LLN_GLI(
  age = people$age_years,
  height = people$height_m,
  gender = people$sex,
  ethnicity = people$ethnicity,
  param = "FVC"
)
lln_ratio <- LLN_GLI(
  age = people$age_years,
  height = people$height_m,
  gender = people$sex,
  ethnicity = people$ethnicity,
  param = "FEV1FVC"
)

diagnosis <- data.frame(
  person = people$person,
  description = people$description,
  pred_fev1_L = pred$pred.FEV1,
  pred_fvc_L = pred$pred.FVC,
  pred_fev1fvc_ratio = pred$pred.FEV1FVC,
  lln_fev1_L = lln_fev1,
  lln_fvc_L = lln_fvc,
  lln_fev1fvc_ratio = lln_ratio
)

print(diagnosis, row.names = FALSE)

a_ratio <- diagnosis$lln_fev1fvc_ratio[diagnosis$person == "A"]
b_ratio <- diagnosis$lln_fev1fvc_ratio[diagnosis$person == "B"]
c_ratio <- diagnosis$lln_fev1fvc_ratio[diagnosis$person == "C"]

check_a_range <- a_ratio >= 0.65 && a_ratio <= 0.68
check_b_higher <- b_ratio > a_ratio
check_volume_mismatch <- any(diagnosis$lln_fev1_L > 1.0) &&
  all(diagnosis$lln_fev1fvc_ratio < 1.0)

cat("\nASSERT-CHECKS\n")
cat(
  "A 65yo male FEV1/FVC ratio LLN ~0.65-0.68:",
  ifelse(check_a_range, "PASS", "FAIL"),
  sprintf("(value=%.6f)\n", a_ratio)
)
cat(
  "B 25yo male ratio LLN higher than A 65yo male:",
  ifelse(check_b_higher, "PASS", "FAIL"),
  sprintf("(B=%.6f, A=%.6f)\n", b_ratio, a_ratio)
)
cat(
  "FEV1 volume LLN is distinct from FEV1/FVC ratio LLN:",
  ifelse(check_volume_mismatch, "PASS", "FAIL"),
  sprintf("(A FEV1 volume LLN=%.6f L, A ratio LLN=%.6f)\n", diagnosis$lln_fev1_L[1], a_ratio)
)

results_dir <- file.path(getwd(), "results")
dir.create(results_dir, showWarnings = FALSE)
write.csv(diagnosis, file.path(results_dir, "gli_diagnosis_reference_values.csv"), row.names = FALSE)
