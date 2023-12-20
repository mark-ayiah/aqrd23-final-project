# Loading libraries
library(tidyverse)
library(caret)
library(rsample)
library(glmnet)
library(glmnetUtils)
library(glue)
library(gt)
library(dplyr)
library(summarytools)
library(gtsummary)
library(modelsummary)
library(pROC)
library(broom)
library(boot)
library(car)

# Load, clean, reformat data
broward_clean <- read_csv("allData/BROWARD_CLEAN.csv")
broward_clean <- broward_clean |>
  rename(charge_degree = `charge_degree (misd/fel)`) |>
  mutate(sex = ifelse(sex == 0, 'male', 'female')) |>
  mutate(race = as.factor(case_when(
    race == 1 ~ "White",
    race == 2 ~ "Black",
    race == 3 ~ "Hispanic",
    race == 4 ~ "Asian",
    race == 5 ~ "Native American",
    race == 6 ~ "Other"
  )))

# Replicating Main Finding | Training 1000 Models --------------------------- 

# Set seed for reproducibility of results
set.seed(438)

# Initializing variables to store results
acc_results_lr7 <- rep(NA, 1000)
acc_results_w_lr7 <- rep(NA, 1000)
acc_results_b_lr7 <- rep(NA, 1000)
fp_results_w_lr7 <- rep(NA, 1000)
fp_results_b_lr7 <- rep(NA, 1000)
fn_results_w_lr7 <- rep(NA, 1000)
fn_results_b_lr7 <- rep(NA, 1000)

acc_results_lr2 <- rep(NA, 1000)
acc_results_w_lr2 <- rep(NA, 1000)
acc_results_b_lr2 <- rep(NA, 1000)
fp_results_w_lr2 <- rep(NA, 1000)
fp_results_b_lr2 <- rep(NA, 1000)
fn_results_w_lr2 <- rep(NA, 1000)
fn_results_b_lr2 <- rep(NA, 1000)

roc_auc_values <- numeric(1000)

conf_matrix2 <- matrix(0, nrow = 2, ncol = 2)
conf_matrix7 <- matrix(0, nrow = 2, ncol = 2)

# train 1000 times on 80/20 test/train split
for (i in 1:1000) {
  # Split the data into training and testing sets
  broward_split <- rsample::initial_split(broward_clean, prop = 0.8)
  broward_train <- training(broward_split)
  broward_test <- testing(broward_split)
  
  
  # define 7 feature and 2 feature logistic regression formulas
  form7 <- as.formula("two_year_recid ~ age + sex + juv_misd_count + 
                     juv_fel_count + priors_count + charge_id + charge_degree")
  lr7 <- glm(form7, data = broward_train, family = binomial)
  lr2 <- glm(two_year_recid ~ age + priors_count, data = broward_train, family = binomial)
  
  # predict on test data
  pred7 <- predict(lr7, newdata = broward_test, type = "response")
  pred2 <- predict(lr2, newdata = broward_test, type = "response")
  predicted_labels7 <- ifelse(pred7 > 0.5, 1, 0)
  predicted_labels2 <- ifelse(pred2 > 0.5, 1, 0)
  conf_matrix_iteration_7 <- table(Actual = broward_test$two_year_recid, Predicted = predicted_labels7)
  conf_matrix_iteration_2 <- table(Actual = broward_test$two_year_recid, Predicted = predicted_labels2)
  conf_matrix7 <- conf_matrix7 + conf_matrix_iteration_7
  conf_matrix2 <- conf_matrix2 + conf_matrix_iteration_2
  
  # Calculate AUC-ROC curve
  roc_curve <- roc(broward_test$two_year_recid, pred7)
  roc_auc_values[i] <- auc(roc_curve)
  
  
  # create dataframes for predictions, split by race
  brow_pred_7 <- broward_test |> 
    mutate(pred = pred7)
  brow_white_7 <- brow_pred_7 |> 
    filter(race == "White")
  brow_black_7 <- brow_pred_7 |> 
    filter(race == "Black")
  
  brow_pred_2 <- broward_test |> 
    mutate(pred = pred2)
  brow_white_2 <- brow_pred_2 |> 
    filter(race == "White")
  brow_black_2 <- brow_pred_2 |> 
    filter(race == "Black")
  
  
  # store model results
  acc_results_lr7[i] <- mean((brow_pred_7$pred > 0.5) == brow_pred_7$two_year_recid)
  acc_results_w_lr7[i] <- mean((brow_white_7$pred > 0.5) == brow_white_7$two_year_recid)
  acc_results_b_lr7[i] <- mean((brow_black_7$pred > 0.5) == brow_black_7$two_year_recid)
  fp_results_w_lr7[i] <- sum((brow_white_7$pred > 0.5) & (brow_white_7$two_year_recid == 0)) /
    sum(brow_white_7$two_year_recid == 0)
  fp_results_b_lr7[i] <- sum((brow_black_7$pred > 0.5) & (brow_black_7$two_year_recid == 0)) /
    sum(brow_black_7$two_year_recid == 0)
  fn_results_w_lr7[i] <- sum((brow_white_7$pred <= 0.5) & (brow_white_7$two_year_recid == 1)) /
    sum(brow_white_7$two_year_recid == 1)
  fn_results_b_lr7[i] <- sum((brow_black_7$pred <= 0.5) & (brow_black_7$two_year_recid == 1)) /
    sum(brow_black_7$two_year_recid == 1)
  
  acc_results_lr2[i] <- mean((brow_pred_2$pred > 0.5) == brow_pred_2$two_year_recid)
  acc_results_w_lr2[i] <- mean((brow_white_2$pred > 0.5) == brow_white_2$two_year_recid)
  acc_results_b_lr2[i] <- mean((brow_black_2$pred > 0.5) == brow_black_2$two_year_recid)
  fp_results_w_lr2[i] <- sum((brow_white_2$pred > 0.5) & (brow_white_2$two_year_recid == 0)) /
    sum(brow_white_2$two_year_recid == 0)
  fp_results_b_lr2[i] <- sum((brow_black_2$pred > 0.5) & (brow_black_2$two_year_recid == 0)) /
    sum(brow_black_2$two_year_recid == 0)
  fn_results_w_lr2[i] <- sum((brow_white_2$pred <= 0.5) & (brow_white_2$two_year_recid == 1)) /
    sum(brow_white_2$two_year_recid == 1)
  fn_results_b_lr2[i] <- sum((brow_black_2$pred <= 0.5) & (brow_black_2$two_year_recid == 1)) /
    sum(brow_black_2$two_year_recid == 1)
}

# Summary Statistics - Tables 1, 7, and 8 ---------------------------

# Table 1 | Total Accuracy ---------------------------

# Defining variables for each statistic's mean and standard deviation
mean_age <- mean(broward_clean$age)
mean_juv_fel_count <- mean(broward_clean$juv_fel_count)
mean_juv_misd_count <- mean(broward_clean$juv_misd_count)
mean_priors_count <- mean(broward_clean$priors_count)
mean_COMPAS_score <- mean(broward_clean$compas_decile_score)

sd_age <- sd(broward_clean$age)
sd_jfc <-sd(broward_clean$juv_fel_count)
sd_jmc <- sd(broward_clean$juv_misd_count)
sd_priors <- sd(broward_clean$priors_count)
sd_COMPAS <- sd(broward_clean$compas_decile_score)

# Aggregating all statistics into one tibble
summary_data <- tibble(
  Variable = c("Age", "Juvenile Felony Count", "Juvenile Misdemeanor Count", "Priors Count", "COMPAS Score"),
  Mean = c(mean_age, mean_juv_fel_count, mean_juv_misd_count, mean_priors_count, mean_COMPAS_score),
  SD = c(sd_age, sd_jfc, sd_jmc, sd_priors, sd_COMPAS)
)

num_observations <- length(na.omit(broward_clean$age))

# Creating final table with formatting
summary_stats_tbl <- summary_data |>
  gt() |>
  tab_spanner(
    label = "Statistics",
    columns = c("Mean", "SD")
  ) |>
  fmt_number(
    columns = c(Mean, SD),
    decimals = 2
  ) |>
  tab_header(
    title = "Defendant Summary Statistics",
    subtitle = "for Broward County Dataset"
  ) |>
  tab_footnote(
    md(paste("Number of observations:    ", num_observations))
  )

# Printing Table 1
print(summary_stats_tbl)

# Save the gt table directly as LaTeX
summary_stats_tex <- gt(as.data.frame(summary_stats_tbl)) %>%
  gt::as_latex()

# Save the LaTeX object to a file
writeLines(summary_stats_tex, "tables/table_1.tex")

# Table 7 | Accuracy for Black Defendants ---------------------------

# Filtering for only Black defendants
broward_clean_b <- broward_clean |>
  filter(race == "Black")

# Repeating same process as table 1
mean_age_b <- mean(broward_clean_b$age)
mean_juv_fel_count_b <- mean(broward_clean_b$juv_fel_count)
mean_juv_misd_count_b <- mean(broward_clean_b$juv_misd_count)
mean_priors_count_b <- mean(broward_clean_b$priors_count)
mean_COMPAS_score_b <- mean(broward_clean_b$compas_decile_score)

sd_age_b <- sd(broward_clean_b$age)
sd_jfc_b <-sd(broward_clean_b$juv_fel_count)
sd_jmc_b <- sd(broward_clean_b$juv_misd_count)
sd_priors_b <- sd(broward_clean_b$priors_count)
sd_COMPAS_b <- sd(broward_clean_b$compas_decile_score)

summary_data_b <- tibble(
  Variable = c("Age", "Juvenile Felony Count", "Juvenile Misdemeanor Count", "Priors Count", "COMPAS Score"),
  Mean = c(mean_age_b, mean_juv_fel_count_b, mean_juv_misd_count_b, mean_priors_count_b, mean_COMPAS_score_b),
  SD = c(sd_age_b, sd_jfc_b, sd_jmc_b, sd_priors_b, sd_COMPAS_b)
)

num_observations_b <- length(na.omit(broward_clean_b$age))


blk_summary_stats_tbl <- summary_data_b |>
  gt() |>
  tab_spanner(
    label = "Statistics",
    columns = c("Mean", "SD")
  ) |>
  fmt_number(
    columns = c(Mean, SD),
    decimals = 2
  ) |>
  tab_header(
    title = "Defendant Summary Statistics",
    subtitle = "for Black Defendants in Broward County Dataset"
  )|> 
  tab_footnote(
    md(paste("Number of observations:    ", num_observations_b))
  )

# Printing Table 7
print(blk_summary_stats_tbl)

# Save the gt table directly as LaTeX
blk_summary_stats_tex <- gt(as.data.frame(blk_summary_stats_tbl)) %>%
  gt::as_latex()

# Save the LaTeX object to a file
writeLines(blk_summary_stats_tex, "tables/table_7.tex")

# Table 8 | Accuracy for White Defendants ---------------------------

# Filtering for only White defendants
broward_clean_w <- broward_clean |>
  filter(race == "White")

# Repeating same process as table 1
mean_age_w <- mean(broward_clean_w$age)
mean_juv_fel_count_w <- mean(broward_clean_w$juv_fel_count)
mean_juv_misd_count_w <- mean(broward_clean_w$juv_misd_count)
mean_priors_count_w <- mean(broward_clean_w$priors_count)
mean_COMPAS_score_w <- mean(broward_clean_w$compas_decile_score)

sd_age_w <- sd(broward_clean_w$age)
sd_jfc_w <-sd(broward_clean_w$juv_fel_count)
sd_jmc_w <- sd(broward_clean_w$juv_misd_count)
sd_priors_w <- sd(broward_clean_w$priors_count)
sd_COMPAS_w <- sd(broward_clean_w$compas_decile_score)

summary_data_w <- tibble(
  Variable = c("Age", "Juvenile Felony Count", "Juvenile Misdemeanor Count", "Priors Count", "COMPAS Score"),
  Mean = c(mean_age_w, mean_juv_fel_count_w, mean_juv_misd_count_w, mean_priors_count_w, mean_COMPAS_score_w),
  SD = c(sd_age_w, sd_jfc_w, sd_jmc_w, sd_priors_w, sd_COMPAS_w)
)

num_observations_w <- length(na.omit(broward_clean_w$age))


wht_summary_stats_tbl <- summary_data_w |>
  gt() |>
  tab_spanner(
    label = "Statistics",
    columns = c("Mean", "SD")
  ) |>
  fmt_number(
    columns = c(Mean, SD),
    decimals = 2
  ) |>
  tab_header(
    title = "Defendant Summary Statistics",
    subtitle = "for Black Defendants in Broward County Dataset"
  )|> 
  tab_footnote(
    md(paste("Number of observations:    ", num_observations_w))
  )

# Printing Table 8
print(wht_summary_stats_tbl)

# Save the gt table directly as LaTeX
wht_summary_stats_tex <- gt(as.data.frame(wht_summary_stats_tbl)) %>%
  gt::as_latex()

# Save the LaTeX object to a file
writeLines(wht_summary_stats_tex, "tables/table_8.tex")



# COMPAS Accuracy | Table 2  ---------------------------

# Loading in and cleaning 1000 person subset data
broward_subset <- read_csv("allData/BROWARD_CLEAN_SUBSET.csv")

broward_subset <- broward_subset |>
  rename(charge_degree = `charge_degree (misd/fel)`) |>
  mutate(sex = ifelse(sex == 0, 'male', 'female'))

broward_subset <- broward_subset |>
  mutate(race = case_when(
    race == 1 ~ "White",
    race == 2 ~ "Black",
    race == 3 ~ "Hispanic",
    race == 4 ~ "Asian",
    race == 5 ~ "Native American",
    race == 6 ~ "Other"
  ))

# Adding variables to indicate COMPAS predictions and correctness
broward_subset$compas_prediction <- ifelse(broward_subset$compas_decile_score > 4, 1, 0) 
broward_subset$compas_correct <- ifelse(broward_subset$compas_prediction == broward_subset$two_year_recid, 1, 0)

# Calculate overall COMPAS accuracy 
accuracy_overall <- broward_subset|>
  summarise(accuracy_overall = sum(compas_correct) / n())

# Creating table with accuracies and false positive/negative rates by race
accuracies <- broward_subset |>
  group_by(race)|>
  summarize(
    race_accuracy = sum(compas_correct) / n(),
    false_pos = sum(compas_correct == 0 & compas_prediction == 1) / (sum(two_year_recid == 0)),
    false_neg = sum(compas_correct == 0 & compas_prediction == 0) / (sum(two_year_recid == 1))
  ) |>
  mutate(
    accuracy_overall = accuracy_overall$accuracy_overall
  )|>
  filter(race %in% c("Black", "White"))


# Finalizing Table 2
accuracy_data <- tibble(
  ` ` = c("Accuracy (overall)", "AUC-ROC (overall)", "Accuracy", "False positive", "False negative"),
  White = c(accuracies$accuracy_overall[2], mean(roc_auc_values), 
            accuracies$race_accuracy[2], accuracies$false_pos[2], accuracies$false_neg[2]),
  Black = c(accuracies$accuracy_overall[1], mean(roc_auc_values),
            accuracies$race_accuracy[1], accuracies$false_pos[1], accuracies$false_neg[1])
)

accuracies_table <- accuracy_data |>
  gt() |>
  tab_spanner(
    label = "Defendant Race",
    columns = c("White", "Black")
  ) |>
  fmt_percent(
    columns = c(White, Black),
    decimals = 1
  ) |>
  tab_header(
    title = "COMPAS algorithmic predictions from 1000 defendants",
    subtitle = "Overall accuracy is specified as percent correct"
  )

# Printing Table 2
print(accuracies_table)

# Save the gt table directly as LaTeX
acc_tex <- gt(as.data.frame(accuracies_table)) %>%
  gt::as_latex()

# Save the LaTeX object to a file
writeLines(acc_tex, "tables/table_2.tex")



# Algorithmic predictions | Table 3 ---------------------------

# Collecting results from 1000 regressions into lists for each model

# 7 Feature Model
lr7_stats <- list(
  acc_results_lr7,
  acc_results_b_lr7,
  acc_results_w_lr7,
  fp_results_b_lr7,
  fp_results_w_lr7,
  fn_results_b_lr7,
  fn_results_w_lr7
)

# 2 Feature Model
lr2_stats <- list(
  acc_results_lr2,
  acc_results_b_lr2,
  acc_results_w_lr2,
  fp_results_b_lr2,
  fp_results_w_lr2,
  fn_results_b_lr2,
  fn_results_w_lr2
)

# Initializing lists for bootstrapped confidence intervals
upper_bounds_7 <- rep(NA, 7)
lower_bounds_7 <- rep(NA, 7)
means_lr7 <- rep(NA, 7)

upper_bounds_2 <- rep(NA, 7)
lower_bounds_2 <- rep(NA, 7)
means_lr2 <- rep(NA, 7)

# Setting seed for reproducibility
set.seed(438)

# Iterate through each metric
for (k in 1:length(lr7_stats)) {
  testing_accuracies_7 <- lr7_stats[[k]]
  testing_accuracies_2 <- lr2_stats[[k]]
  num_bootstrap_samples <- 100
  num_resamples <- 1000
  resampled_means_7 <- numeric(num_resamples)
  resampled_means_2 <- numeric(num_resamples)
  
  # Perform bootstrapping for both models
  for (i in 1:num_resamples) {
    # Resample with replacement from the testing accuracies
    resample_accuracies_7 <- sample(testing_accuracies_7, replace = TRUE)
    resample_accuracies_2 <- sample(testing_accuracies_2, replace = TRUE)
    
    # Calculate the mean of the resampled accuracies
    resampled_mean_7 <- mean(resample_accuracies_7)
    resampled_mean_2 <- mean(resample_accuracies_2)
    
    # Store the resampled mean
    resampled_means_7[i] <- resampled_mean_7
    resampled_means_2[i] <- resampled_mean_2
  }
  
  # Calculate the mean of the resampled means
  mean_of_resampled_means_7 <- mean(resampled_means_7)
  mean_of_resampled_means_2 <- mean(resampled_means_2)
  
  # Calculate the bootstrapped confidence interval
  lower_bounds_7[k] <- quantile(resampled_means_7, 0.025)
  upper_bounds_7[k] <- quantile(resampled_means_7, 0.975)
  means_lr7[k] <- mean_of_resampled_means_7
  lower_bounds_2[k] <- quantile(resampled_means_2, 0.025)
  upper_bounds_2[k] <- quantile(resampled_means_2, 0.975)
  means_lr2[k] <- mean_of_resampled_means_2
}

# Creating and formatting Table 3
model_tags <- c(rep("(A) LR7", 7), rep("(B) LR2", 7))
lower_bds <- c(lower_bounds_7*100, lower_bounds_2*100)
upper_bds <- c(upper_bounds_7*100, upper_bounds_2*100)
mean_stats <- c(means_lr7*100, means_lr2*100)
var_names <- c("Accuracy", "Accuracy (Black)", "Accuracy (White)", 
               "False Positive Rate (Black)", 
               "False Positive Rate (White)", 
               "False Negative Rate (Black)", 
               "False Negative Rate (White)", 
               "Accuracy", "Accuracy (Black)", "Accuracy (White)", 
               "False Positive Rate (Black)", 
               "False Positive Rate (White)", 
               "False Negative Rate (Black)", 
               "False Negative Rate (White)")

ci_table <- tibble(` ` = var_names,
                   model = model_tags,
                   mean = round(mean_stats, 2),
                   lower.bound = round(lower_bds, 2),
                   upper.bound = round(upper_bds, 2))
ci_table <- ci_table |> 
  mutate(val = paste0(as.character(mean), "%  [", 
                      as.character(lower.bound), ", ", 
                      as.character(upper.bound), "]")) |> 
  select(` `, model, val) |> 
  pivot_wider(names_from = model, values_from = val) |> 
  gt() |> 
  tab_header(
    title = md("**Algorithmic predicitions from 7214 defendants**"),
    subtitle = "Logistic regression with 7 features (A) (LR7), 
    logistic regression with 2 features (B) (LR2). The values in the square brackets correspond 
    to the 95% bootstrapped confidence intervals."
  )

# Printing Table 3
print(ci_table)

# Save the gt table directly as LaTeX
ci_tex <- gt(as.data.frame(ci_table)) %>%
  gt::as_latex()

# Save the LaTeX object to a file
writeLines(ci_tex, "tables/table_3.tex")



# Performance metrics | Table 4 ---------------------------

# Creating tibble with corresponding metrics
metrics <- tibble(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score", "False Negative", "False Positive"),
  "(A) Seven Feature" = c( 0.67764, 0.673229, 0.540458, 0.59958, 0.459542, 0.215152),
  "(B) Two Feature" = c(0.6747, 0.688198,  0.520356, 0.592622, 0.47964443, 0.19336)
)

# Formatting table in gt
metrics_table <- metrics |>
  gt() |>
  tab_spanner(
    label = "Classifier",
    columns = c("(A) Seven Feature", "(B) Two Feature")
  ) |>
  fmt_number(
    columns = c("(A) Seven Feature", "(B) Two Feature"),
    decimals = 3
  ) |>
  tab_header(
    title = "Performance metrics for 7 feature and 2 feature classifiers",
    subtitle = "Based on results from training 1000 times on 80% training 20% testing split"
  )

# Printing Table 4
print(metrics_table)

# Save the gt table directly as LaTeX
metrics_tex <- gt(as.data.frame(metrics_table)) %>%
  gt::as_latex()

# Save the LaTeX object to a file
writeLines(metrics_tex, "tables/table_4.tex")



# Precision Tables | Tables 5 & 6 ---------------------------

# Creating precision table for 7 feature model
precision_df7 <- tibble(
  Classification = c("Actually No", "Actually Yes"),
  `Predicted No` = c(conf_matrix7[1,1], conf_matrix7[2,1]),
  `Predicted Yes` = c(conf_matrix7[1,2], conf_matrix7[2,2])
)

# Formatting precision table
precision_table7 <- precision_df7 |>
  gt() |>
  tab_spanner(
    label = "2 Feature Classifier Prediction",
    columns = c("Predicted No", "Predicted Yes")
  ) |>
  fmt_number(
    columns = c(`Predicted No`, `Predicted Yes`),
    decimals = 0) |>
  tab_header(
    title = "7 Feature Classifier Prediction of Recidivism vs Actual Recidivism",
    subtitle = "Results from training 1000 times on defendant data with features age & prior count",
  ) 
# Printing Table 5
print(precision_table7)

# Save the gt table directly as LaTeX
prec7_tex <- gt(as.data.frame(precision_table7)) %>%
  gt::as_latex()

# Save the LaTeX object to a file
writeLines(prec7_tex, "tables/table_5.tex")


# Creating precision table for 2 feature model
precision_df2 <- tibble(
  Classification = c("Actually No", "Actually Yes"),
  `Predicted No` = c(conf_matrix2[1,1], conf_matrix2[2,1]),
  `Predicted Yes` = c(conf_matrix2[1,2], conf_matrix2[2,2])
)

# Formatting precision table
precision_table2 <- precision_df2 |>
  gt() |>
  tab_spanner(
    label = "2 Feature Classifier Prediction",
    columns = c("Predicted No", "Predicted Yes")
  ) |>
  fmt_number(
    columns = c(`Predicted No`, `Predicted Yes`),
    decimals = 0) |>
  tab_header(
    title = "2 Feature Classifier Prediction of Recidivism vs Actual Recidivism",
    subtitle = "Results from training 1000 times on defendant data with features age & prior count",
  ) 
# Printing Table 6
print(precision_table2)

# Save the gt table directly as LaTeX
prec2_tex <- gt(as.data.frame(precision_table2)) %>%
  gt::as_latex()

# Save the LaTeX object to a file
writeLines(prec2_tex, "tables/table_6.tex")
