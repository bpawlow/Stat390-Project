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







svmpoly_model <- svm_poly(
  mode = "regression",
  cost = tune(),
  degree = tune(),
) %>%
  set_engine("kernlab")

# set-up tuning grid ----
svmpoly_params <- hardhat::extract_parameter_set_dials(svmpoly_model)

# define grid
svmpoly_grid <- grid_regular(svmpoly_params, levels = 5)

# workflow ----
svmpoly_workflow <- workflow() %>%
  add_model(svmpoly_model) %>%
  add_recipe(recipe)

# Tuning/fitting ----
svmpoly_tuned <- svmpoly_workflow %>%
  tune_grid(
    resamples = folds,
    grid = svmpoly_grid,
    control = ctrl_grid
  )


write_rds(svmpoly_tuned, "model selection/model_objects/svmpoly_tuned.rds")


svmpoly_workflow_tuned <- svmpoly_workflow %>%
  finalize_workflow(select_best(svmpoly_tuned, metric = "rmse"))






svmpoly_fit_folds <- fit_resamples(
  svmpoly_workflow_tuned, 
  resamples = folds, 
  control = control_grid(save_pred = T))



svmpoly_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )





total_model_results <- (svmpoly_fit_folds %>% collect_metrics() %>% mutate(model = "svmpoly"))

total_model_results %>% 
  filter(.metric == "rmse") %>% 
  arrange(mean) 




recipe_pca <- recipe(y ~ .,
                 pca_train) %>%
  step_rm(id) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) 


peek_pca <- recipe_pca %>% prep() %>% bake(pca_train)







svmpoly_model_pca <- svm_poly(
  mode = "regression",
  cost = tune(),
  degree = tune(),
) %>%
  set_engine("kernlab")

# set-up tuning grid ----
svmpoly_params_pca <- hardhat::extract_parameter_set_dials(svmpoly_model_pca)

# define grid
svmpoly_grid_pca <- grid_regular(svmpoly_params_pca, levels = 5)

# workflow ----
svmpoly_workflow_pca <- workflow() %>%
  add_model(svmpoly_model_pca) %>%
  add_recipe(recipe_pca)

# Tuning/fitting ----
svmpoly_tuned_pca <- svmpoly_workflow_pca %>%
  tune_grid(
    resamples = folds_pca,
    grid = svmpoly_grid_pca,
    control = ctrl_grid
  )


write_rds(svmpoly_tuned_pca, "model selection/model_objects/svmpoly_tuned_pca.rds")


svmpoly_workflow_tuned_pca <- svmpoly_workflow_pca %>%
  finalize_workflow(select_best(svmpoly_tuned, metric = "rmse"))






svmpoly_fit_folds_pca <- fit_resamples(
  svmpoly_workflow_tuned_pca, 
  resamples = folds_pca, 
  control = control_grid(save_pred = T))



svmpoly_fit_folds_pca %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )





svmpoly_fit_folds_pca %>% 
  collect_metrics() %>% 
  mutate(model = "svmpoly") %>%
  total_model_results %>% 
  filter(.metric == "rmse") 
