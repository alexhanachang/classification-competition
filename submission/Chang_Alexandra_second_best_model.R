#### log model ####
# load-packages -----------------------------------------------------------
library(glmnet)
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
log_model <- logistic_reg(
  penalty = tune(), 
  mixture = tune()
  ) %>% 
  set_mode("classification") %>% 
  set_engine("glmnet")


# define-tuning-grid ------------------------------------------------------
log_params <- parameters(log_model)

log_grid <- grid_regular(
  penalty(), 
  mixture(), 
  levels = 5
)


# define-workflow ---------------------------------------------------------
log_wflow <-
  workflow() %>% 
  add_model(log_model) %>% 
  add_recipe(recipe)


# tuning ------------------------------------------------------------------
log_tuned <- log_wflow %>% 
  tune_grid(fold, grid = log_grid)

log_tuned %>% 
  show_best(metric = "accuracy")

write_rds(log_tuned, "model_info/log_tuned.rds")


# tuned-workflow -------------------------------------------------------
log_wflow_tuned <- log_wflow %>%
  finalize_workflow(select_best(log_tuned, metric = "accuracy"))


# fit ---------------------------------------------------------------------
log_results <- fit(log_wflow_tuned, train)


# test-set-performance ----------------------------------------------------
log_predictions <- log_results %>%
  predict(new_data = test) %>% 
  bind_cols(test %>% select(id)) %>%  
  select(id, .pred_class) %>% 
  rename(Id = id, Category = .pred_class)

write_csv(log_predictions, "predictions/log_predictions.csv")




