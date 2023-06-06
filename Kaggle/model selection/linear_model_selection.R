library(tidyverse)
library(tidymodels)
library(doMC)
parallel::detectCores()
registerDoMC(cores = 10)

set.seed(3013)

fv <-
  c(
    'x105',
    'x102',
    'x561',
    'x702',
    'x696',
    'x567',
    'x111',
    'x369',
    'x516',
    'x654',
    'x685',
    'x591',
    'x585',
    "x619", 
    'x118', 
    'x652', 
    'x114', 
    'x358',
    'x366',
    'x506',
    'x532',
    'x668',
    'x168'
  )



train <- read_csv("data/train_cleaned.csv") %>% mutate(x516 = as_factor(x516))

pca_train <- read_csv('data/pca_train.csv') %>% mutate(x516 = as_factor(x516))

folds <- vfold_cv(train, v = 5, repeats = 3)


folds_pca <-  vfold_cv(pca_train, v = 5, repeats = 3)


recipe <- recipe(y ~ .,
                 train) %>%
  step_rm(id) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) 


peek <- recipe %>% prep() %>% bake(train)


lm_model <- linear_reg() %>%
  set_engine("lm")

lm_workflow <- workflow() %>%
  add_model(lm_model) %>%
  add_recipe(recipe)


lm_fit_folds <- fit_resamples(
  lm_workflow, 
  resamples = folds, 
  control = control_grid(save_pred = T))



lm_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )




total_model_results <- (lm_fit_folds %>% collect_metrics() %>% mutate(model = "Lasso"))

total_model_results %>% 
  filter(.metric == "rmse")





recipe_pca <- recipe(y ~ .,
                     pca_train) %>%
  step_rm(id) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) 


peek_pca <- recipe %>% prep() %>% bake(pca_train)

lm_model_pca <- linear_reg() %>%
  set_engine("lm")

lm_workflow_pca <- workflow() %>%
  add_model(lm_model_pca) %>%
  add_recipe(recipe_pca)


lm_fit_folds_pca <- fit_resamples(
  lm_workflow_pca, 
  resamples = folds_pca, 
  control = control_grid(save_pred = T))



lm_fit_folds_pca %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )




total_model_results_pca <- (lm_fit_folds_pca %>% collect_metrics() %>% mutate(model = "Lasso"))

total_model_results_pca %>% 
  filter(.metric == "rmse")






