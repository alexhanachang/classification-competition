# load packages
library(ggplot2)
library(glmnet)
library(janitor)
library(patchwork)
library(skimr)
library(tidymodels)
library(tidyverse)
library(yardstick)

# set seed
set.seed(30102)

# read-in
finaldata <- readRDS("data/processed/finaldata.rds")

# data split
split <- initial_split(finaldata, prop = 0.75, strata = outcome)

# train data set
train <- training(split)

# test data set
test <- testing(split)

# folded data set
folds <- vfold_cv(data = train, v = 10, repeats = 5)
folds

#### recipe ####
# recipe()
recipe <- recipe(outcome ~ ., data = train) %>%
  step_impute_linear(age) %>% # step_impute_linear()
  step_dummy(sexual_identity, # dummy variables
             intersex, gender_identity, trans, assigned_birth,
             race, diagnosed_hiv,
             starts_with("fin_"),  
             starts_with("soc_"), 
             starts_with("phys_"),
             starts_with("risk"), 
             one_hot = T) %>% # one_hot encode 
  step_normalize(all_predictors()) 

# prep() and bake()
recipe %>% 
  prep() %>% 
  bake(new_data = NULL)

# logisitc model
log_model <- 
  logistic_reg(
    penalty = tune(), 
    mixture = tune()
  ) %>% 
  set_mode("classification") %>% 
  set_engine("glmnet")

# parameters
log_params <- parameters(log_model)
log_params

log_grid <- grid_regular(
  penalty(), 
  mixture(), 
  levels = 5
)
log_grid

# workflow
log_wf <-
  workflow() %>% 
  add_model(log_model) %>% 
  add_recipe(recipe)

# tuning
log_tuned <-
  log_wf %>% 
  tune_grid(folds, grid = log_grid)

# collect metrics
log_tuned %>% 
  collect_metrics() %>% 
  arrange(- mean)
## # A tibble: 50 x 8
##         penalty mixture .metric .estimator  mean     n std_err .config              
##           <dbl>   <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>                
##  1 1               0    roc_auc  binary     0.732    50 0.00854 Preprocessor1_Model05
##  2 0.00316         1    accuracy binary     0.715    50 0.00714 Preprocessor1_Model24
##  3 0.00316         0.25 accuracy binary     0.715    50 0.00677 Preprocessor1_Model09
##  4 0.0000000001    0.75 accuracy binary     0.715    50 0.00675 Preprocessor1_Model16
##  5 0.0000000316    0.75 accuracy binary     0.715    50 0.00675 Preprocessor1_Model17
##  6 0.00001         0.75 accuracy binary     0.715    50 0.00675 Preprocessor1_Model18
##  7 0.0000000001    1    accuracy binary     0.715    50 0.00675 Preprocessor1_Model21
##  8 0.0000000316    1    accuracy binary     0.715    50 0.00675 Preprocessor1_Model22
##  9 0.00001         1    accuracy binary     0.715    50 0.00675 Preprocessor1_Model23
## 10 0.0000000001    0.5  accuracy binary     0.714    50 0.00677 Preprocessor1_Model11
## # … with 40 more rows

log_tuned %>% 
  select_best(metric = "roc_auc")

log_tuned %>% 
  select_best(metric = "accuracy")

# save 
write_rds(log_tuned, "models/log_results.rds")


