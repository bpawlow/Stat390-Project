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


knn_model <- nearest_neighbor(
  mode = "regression",
  neighbors = tune()
) %>%
  set_engine("kknn")



# set-up tuning grid ----
knn_params <- hardhat::extract_parameter_set_dials(knn_model) %>%
  update(neighbors = neighbors(range = c(1,100)))

# define grid
knn_grid <- grid_regular(knn_params, levels = 15)

# workflow ----
knn_workflow <- workflow() %>%
  add_model(knn_model) %>%
  add_recipe(recipe)

# Tuning/fitting ----
knn_tuned <- knn_workflow %>%
  tune_grid(folds, 
            grid = knn_grid,
            control = ctrl_grid
            )

write_rds(knn_tuned, "model selection/model_objects/knn_tuned.rds")


knn_workflow_tuned <- knn_workflow %>%
  finalize_workflow(select_best(knn_tuned, metric = "rmse"))



knn_fit_folds <- fit_resamples(
  knn_workflow_tuned, 
  resamples = folds, 
  control = control_grid(save_pred = T))



knn_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )




total_model_results <- (knn_fit_folds %>% collect_metrics() %>% mutate(model = "knn"))

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


knn_model_pca <- nearest_neighbor(
  mode = "regression",
  neighbors = tune()
) %>%
  set_engine("kknn")



# set-up tuning grid ----
knn_params_pca <- hardhat::extract_parameter_set_dials(knn_model_pca) %>%
  update(neighbors = neighbors(range = c(1,100)))

# define grid
knn_grid_pca <- grid_regular(knn_params_pca, levels = 15)

# workflow ----
knn_workflow_pca <- workflow() %>%
  add_model(knn_model_pca) %>%
  add_recipe(recipe_pca)

# Tuning/fitting ----
knn_tuned_pca <- knn_workflow_pca %>%
  tune_grid(folds_pca, 
            grid = knn_grid_pca,
            control = ctrl_grid
            )

write_rds(knn_tuned_pca, "model selection/model_objects/knn_tuned_pca.rds")



knn_workflow_tuned_pca <- knn_workflow_pca %>%
  finalize_workflow(select_best(knn_tuned_pca, metric = "rmse"))



knn_fit_folds_pca <- fit_resamples(
  knn_workflow_tuned_pca, 
  resamples = folds_pca, 
  control = control_grid(save_pred = T))



knn_fit_folds_pca %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )




total_model_results_pca <- (knn_fit_folds_pca %>% collect_metrics() %>% mutate(model = "knn"))

total_model_results_pca %>% 
  filter(.metric == "rmse")
















