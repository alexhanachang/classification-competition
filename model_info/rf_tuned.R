#### random forest model ####
# load-packages -----------------------------------------------------------
library(janitor)
library(skimr)
library(tidymodels)
library(tidyverse)


# set-seed -----------------------------------------------------------------
set.seed(2021)


# read-in-data ------------------------------------------------------------
train <- read_csv(file = "data/train.csv") %>% 
  clean_names() %>% 
  mutate(hi_int_prncp_pd = factor(hi_int_prncp_pd, levels = c("1", "0")))

test <- read_csv(file = "data/test.csv") %>%
  clean_names()

fold <- vfold_cv(train, v = 5, repeats = 3, strata = hi_int_prncp_pd)

train %>% skim_without_charts()
#character: 
#  addr_state, application_type, earliest_cr_line, emp_length, emp_title, grade, 
#home_owndership, initial_list_status, last_credit_pull_d, purpose, sub_grade, term, verification_status

#numeric:
#  id, acc_now_delinq, acc_open_past_24mths, annual_inc, avg_cur_bal, 
#bc_util, delinq_2yrs, delinq_amnt, dti, int_rate, loan_amnt, mort_acc, 
#num_sats, num_tl_120dpd_2m, num_tl_90g_dpd_24m, num_tl_30dpd, out_prncp_inv, 
#pub_rec, pub_rec_bankruptcies, tot_coll_amt, tot_cur_bal, total_rec_late_fee, 


# recipe ------------------------------------------------------------------
recipe <- recipe(hi_int_prncp_pd ~ ., data = train) %>% 
  step_rm(id, emp_length, earliest_cr_line, last_credit_pull_d, purpose) %>% 
  step_other(emp_title, addr_state, sub_grade, threshold = 0.05) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_nzv(all_predictors())

recipe %>%
  prep() %>%
  bake(new_data = NULL)


# define-model ------------------------------------------------------------
rf_model <- rand_forest(
  mode = "classification",
  mtry = tune(),
  min_n = tune()) %>% 
  set_engine("ranger")


# define-tuning-grid ------------------------------------------------------
rf_params <- parameters(rf_model) %>% finalize(fold)

rf_grid <- grid_regular(rf_params, levels = 3) 


# define-workflow ---------------------------------------------------------
rf_wflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(recipe)


# tuning ------------------------------------------------------------------
rf_tuned <- rf_wflow %>% 
  tune_grid(resamples = fold, grid = rf_grid)

show_best(rf_tuned, metric = "accuracy")

write_rds(rf_tuned, "model_info/rf_tuned.rds")


# tuned-workflow -------------------------------------------------------
rf_wflow_tuned <- rf_wflow %>%
  finalize_workflow(select_best(rf_tuned, metric = "accuracy"))


# fit ---------------------------------------------------------------------
rf_results <- fit(rf_wflow_tuned, train)


# test-set-performance ----------------------------------------------------
rf_predictions <- rf_results %>%
  predict(new_data = test) %>% 
  bind_cols(test %>% select(id)) %>% 
  select(id, .pred_class) %>% 
  rename(Id = id, Category = .pred_class)

write_csv(rf_predictions, "predictions/rf_predictions.csv")


