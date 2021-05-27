# load-packages -----------------------------------------------------------
library(glmnet)
library(lubridate)
library(skimr)
library(tidymodels)
library(tidyverse)


# read-in-data ------------------------------------------------------------
test <- read.csv(file = "data/test.csv") 
train <- read.csv(file = "data/train.csv")

train <- train %>% 
  mutate_if(is_character, as_factor) %>% 
  mutate(
    hi_int_prncp_pd = factor(hi_int_prncp_pd, levels = c("1", "0"))
  )


# set-seed -----------------------------------------------------------------
set.seed(2021)


# quality-check -----------------------------------------------------------
skim_without_charts(train)


# recipe ------------------------------------------------------------------
recipe <- recipe(hi_int_prncp_pd ~ ., data = train) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  step_interact(hi_int_prncp_pd ~ (.)^2) %>% 
  step_other(all_nominal(), -all_outcomes(), threshold = 0.2) %>% 
  step_normalize(all_predictors()) %>% 
  step_zv(all_predictors()) 
  

recipe %>% 
  prep() %>% 
  bake(new_data = NULL)


# fold-data ---------------------------------------------------------------
folds <- vfold_cv(data = train, v = 5, repeats = 3, strata = hi_int_prncp_pd)


# define-model ------------------------------------------------------------
rf_model <- rand_forest(
  mtry = tune(), 
  min_n = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("ranger")


# define-tuning-grid ------------------------------------------------------
rf_params <- parameters(rf_model) %>% 
  update(
    mtry = mtry(range = c(2, 10))
  )

rf_grid <- grid_regular(rf_params, levels = 5)


# define-workflow ---------------------------------------------------------
rf_wflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(rf_recipe)


# tuning ------------------------------------------------------------------
rf_tuned <- rf_wflow %>% 
  tune_grid(folds, grid = rf_grid)



