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

mars_model <- mars(
  num_terms = tune(),
  prod_degree = tune(),
  prune_method = tune(),
  mode = 'regression'
)

# set-up tuning grid ----
mars_params <- hardhat::extract_parameter_set_dials(mars_model)

# define grid
mars_grid <- grid_regular(mars_params, levels = 5) 

# workflow ----
mars_workflow <- workflow() %>%
  add_model(mars_model) %>%
  add_recipe(recipe)

# Tuning/fitting ----
mars_tuned <- mars_workflow %>%
  tune_grid(
    resamples = folds,
    grid = mars_grid,
    control = ctrl_grid
  )

write_rds(mars_tuned, "model selection/model_objects/mars_tuned.rds")

mars_workflow_tuned <- mars_workflow %>%
  finalize_workflow(select_best(mars_tuned, metric = "rmse"))


mars_fit_folds <- fit_resamples(
  mars_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T)
  )


mars_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )


total_model_results <- (mars_fit_folds %>% collect_metrics() %>% mutate(model = "Mars"))

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

mars_model_pca <- mars(
  num_terms = tune(),
  prod_degree = tune(),
  prune_method = tune(),
  mode = 'regression'
)

# set-up tuning grid ----
mars_params_pca <- hardhat::extract_parameter_set_dials(mars_model_pca)

# define grid
mars_grid_pca <- grid_regular(mars_params_pca, levels = 5) 

# workflow ----
mars_workflow_pca <- workflow() %>%
  add_model(mars_model_pca) %>%
  add_recipe(recipe_pca)

# Tuning/fitting ----
mars_tuned_pca <- mars_workflow_pca %>%
  tune_grid(
    resamples = folds_pca,
    grid = mars_grid_pca,
    control = ctrl_grid
  )
write_rds(mars_tuned_pca, "model selection/model_objects/mars_tuned_pca.rds")


mars_workflow_tuned_pca <- mars_workflow_pca %>%
  finalize_workflow(select_best(mars_tuned_pca, metric = "rmse"))


mars_fit_folds_pca <- fit_resamples(
  mars_workflow_tuned_pca, 
  resamples = folds_pca,
  control = control_grid(save_pred = T)
)


mars_fit_folds_pca %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )


total_model_results_pca <- (mars_fit_folds_pca %>% collect_metrics() %>% mutate(model = "Mars"))

total_model_results_pca %>% 
  filter(.metric == "rmse") %>% 
  arrange(mean) 

