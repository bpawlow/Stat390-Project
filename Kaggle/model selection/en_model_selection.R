library(tidyverse)
library(tidymodels)
library(doMC)
library(rules)
parallel::detectCores()
registerDoMC(cores = 10)
tidymodels_prefer()
conflicted::conflicts_prefer(scales::alpha)
conflicted::conflicts_prefer(recipes::update)

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



train <-
  read_csv("data/train_cleaned.csv") %>% mutate(x516 = as_factor(x516))
pca_train <-
  read_csv('data/pca_train.csv') %>% mutate(x516 = as_factor(x516))
folds <- read_rds('model selection/model_objects/folds.rds')
folds_pca <-
  read_rds('model selection/model_objects/folds_pca.rds')
ctrl_grid <- read_rds("model selection/model_objects/ctrl_grid.rds")



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

# Define model ----
en_model <- linear_reg(
  mixture = tune(),
  penalty = tune(),
  mode = "regression",
) %>%
  set_engine("glmnet")

en_params <- hardhat::extract_parameter_set_dials(en_model)

# define grid
en_grid <- grid_regular(en_params, levels = 5)

# workflow ----
en_workflow <- workflow() %>%
  add_model(en_model) %>%
  add_recipe(recipe)

# Tuning/fitting ----
en_tuned <- en_workflow %>%
  tune_grid(
    resamples = folds,
    grid = en_grid,
    control = ctrl_grid)

write_rds(en_tuned, "model selection/model_objects/en_tuned.rds")



en_workflow_tuned <- en_workflow %>%
  finalize_workflow(select_best(en_tuned, metric = "rmse"))



en_fit_folds <- fit_resamples(
  en_workflow_tuned, 
  resamples = folds, 
  control = control_grid(save_pred = T))



en_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )




total_model_results <- (en_fit_folds %>% collect_metrics() %>% mutate(model = "knn"))

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


# Define model ----
en_model_pca <- linear_reg(
  mixture = tune(),
  penalty = tune(),
  mode = "regression",
) %>%
  set_engine("glmnet")

en_params_pca <- hardhat::extract_parameter_set_dials(en_model_pca)

# define grid
en_grid_pca <- grid_regular(en_params_pca, levels = 5)

# workflow ----
en_workflow_pca <- workflow() %>%
  add_model(en_model_pca) %>%
  add_recipe(recipe_pca)

# Tuning/fitting ----
en_tuned_pca <- en_workflow_pca %>%
  tune_grid(
    resamples = folds_pca,
    grid = en_grid_pca,
    control = ctrl_grid)

write_rds(en_tuned_pca, "model selection/model_objects/en_tuned_pca.rds")



en_workflow_tuned_pca <- en_workflow_pca %>%
  finalize_workflow(select_best(en_tuned_pca, metric = "rmse"))



en_fit_folds_pca <- fit_resamples(
  en_workflow_tuned_pca, 
  resamples = folds_pca, 
  control = control_grid(save_pred = T))



en_fit_folds_pca %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )




total_model_results_pca <- (en_fit_folds_pca %>% collect_metrics() %>% mutate(model = "knn"))

total_model_results_pca %>% 
  filter(.metric == "rmse")




