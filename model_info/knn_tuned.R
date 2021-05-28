#### knn model ####
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
knn_model <- nearest_neighbor(
  mode = "classification", 
  neighbors = tune()
) %>% 
  set_engine("kknn")


# define-tuning-grid ------------------------------------------------------
knn_params <- parameters(knn_model)
knn_params

knn_grid <- grid_regular(knn_params, levels = 5)
knn_grid


# define-workflow ------------------------------------------------------
knn_wflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(recipe)


# tuning ------------------------------------------------------------------
knn_tuned <- knn_wflow %>% 
  tune_grid(fold, grid = knn_grid)

knn_tuned %>% 
  show_best(metric = "accuracy")

write_rds(knn_tuned, "model_info/knn_tuned.rds")


# tuned-workflow -------------------------------------------------------
knn_wflow_tuned <- knn_wflow %>%
  finalize_workflow(select_best(knn_tuned, metric = "accuracy"))


# fit ---------------------------------------------------------------------
knn_results <- fit(knn_wflow_tuned, train)


# test-set-performance ----------------------------------------------------
knn_predictions <- knn_results %>%
  predict(new_data = test) %>% 
  bind_cols(test %>% select(id)) %>%  
  select(id, .pred_class) %>% 
  rename(Id = id, Category = .pred_class)

write_csv(knn_predictions, "predictions/knn_predictions.csv")




