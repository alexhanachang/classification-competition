#### KNN ####
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

train %>% glimpse()
# set-seed -----------------------------------------------------------------
set.seed(2021)


# quality-check -----------------------------------------------------------
skim_without_charts(train)


# recipe ------------------------------------------------------------------
recipe <- recipe(hi_int_prncp_pd ~ ., data = train) %>% 
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  step_normalize(all_numeric()) 

recipe %>% 
  prep() %>% 
  bake(new_data = NULL)


# fold-data ---------------------------------------------------------------
folds <- vfold_cv(data = train, v = 5, repeats = 3, strata = hi_int_prncp_pd)


# define-model ------------------------------------------------------------
knn_model <- nearest_neighbor(
  neighbors = tune()
) %>% 
  set_engine("kknn") %>% 
  set_mode("classification")


# define-tuning-grid ------------------------------------------------------
knn_params <- parameters(knn_model)
knn_grid <- grid_regular(knn_params, levels = 5)


# workflow ----------------------------------------------------------------
knn_wflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(recipe)



knn_tuned <- knn_wflow %>% 
  tune_grid(folds, grid = knn_grid)



write_rds(knn_tuned, "model_info/knn_results.rds")





