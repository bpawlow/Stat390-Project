library(tidyverse)
library(tidymodels)
library(doMC)
library(baguette)
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



train <- read_csv("data/train_cleaned.csv") %>% mutate(x516 = as_factor(x516))
pca_train <- read_csv('data/pca_train.csv') %>% mutate(x516 = as_factor(x516))
folds <- read_rds('model selection/model_objects/folds.rds')
folds_pca <-  read_rds('model selection/model_objects/folds_pca.rds')
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

bag_tree_model <- bag_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune(),
  mode = 'regression'
)

# set-up tuning grid ----
bag_tree_params <- hardhat::extract_parameter_set_dials(bag_tree_model)

# define grid
bag_tree_grid <- grid_regular(bag_tree_params, levels = 5) 

# workflow ----
bag_tree_workflow <- workflow() %>%
  add_model(bag_tree_model) %>%
  add_recipe(recipe)

# Tuning/fitting ----
bag_tree_tuned <- bag_tree_workflow %>%
  tune_grid(
    resamples = folds,
    grid = bag_tree_grid,
    control = ctrl_grid
  )


write_rds(bag_tree_tuned, "model selection/model_objects/bag_tree_tuned.rds")


bag_tree_workflow_tuned <- bag_tree_workflow %>%
  finalize_workflow(select_best(bag_tree_tuned, metric = "rmse"))



bag_tree_fit_folds <- fit_resamples(
  bag_tree_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T)
)




bag_tree_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )


bag_tree_fit_folds %>%
  collect_metrics() %>%
  mutate(model = "bmlp") %>% 
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

bag_tree_model_pca <- bag_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune(),
  mode = 'regression'
)

# set-up tuning grid ----
bag_tree_params_pca <- hardhat::extract_parameter_set_dials(bag_tree_model_pca)

# define grid
bag_tree_grid_pca <- grid_regular(bag_tree_params_pca, levels = 5) 

# workflow ----
bag_tree_workflow_pca <- workflow() %>%
  add_model(bag_tree_model_pca) %>%
  add_recipe(recipe_pca)

# Tuning/fitting ----
bag_tree_tuned_pca <- bag_tree_workflow_pca %>%
  tune_grid(
    resamples = folds_pca,
    grid = bag_tree_grid_pca,
    control = ctrl_grid
  )


write_rds(bag_tree_tuned_pca, "model selection/model_objects/bag_tree_tuned_pca.rds")


bag_tree_workflow_tuned_pca <- bag_tree_workflow_pca %>%
  finalize_workflow(select_best(bag_tree_tuned_pca, metric = "rmse"))



bag_tree_fit_folds_pca <- fit_resamples(
  bag_tree_workflow_tuned_pca, 
  resamples = folds_pca,
  control = control_grid(save_pred = T)
)





bag_tree_fit_folds_pca %>%
  collect_predictions() %>%
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y))  %>%
  rmse(truth = y,
       estimate = .pred)


bag_tree_fit_folds_pca %>%
  collect_metrics() %>%
  mutate(model = "bag tree") %>% 
  filter(.metric == "rmse")

