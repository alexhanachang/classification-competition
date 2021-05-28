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
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(c(1, 10)))

rf_grid <- grid_regular(rf_params, levels = 3) 


# define-workflow ---------------------------------------------------------
rf_wflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(recipe)


# tuning ------------------------------------------------------------------
rf_tuned2 <- rf_wflow %>% 
  tune_grid(resamples = fold, grid = rf_grid)

show_best(rf_tuned2, metric = "accuracy")

write_rds(rf_tuned2, "model_info/rf_tuned2.rds")


# tuned-workflow -------------------------------------------------------
rf_wflow_tuned <- rf_wflow %>%
  finalize_workflow(select_best(rf_tuned2, metric = "accuracy"))


# fit ---------------------------------------------------------------------
rf_results <- fit(rf_wflow_tuned, train)


# test-set-performance ----------------------------------------------------
rf_predictions <- rf_results %>%
  predict(new_data = test) %>% 
  bind_cols(test %>% select(id)) %>%  
  select(id, .pred_class) %>% 
  rename(Id = id, Category = .pred_class)

write_csv(rf_predictions, "predictions/rf_predictions2.csv")


