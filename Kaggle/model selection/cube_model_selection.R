library(tidyverse)
library(tidymodels)
library(doMC)
library(rules)
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



train <- read_csv("data/train_cleaned.csv")  %>% mutate(x516 = as_factor(x516))

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


cube_model <- parsnip::cubist_rules(
  committees = tune(),
  neighbors = tune()
)



# set-up tuning grid ----
cube_params <- hardhat::extract_parameter_set_dials(cube_model) %>%
  update(neighbors = neighbors(c(5,40)),
         committees = committees(c(3,40))
  )

# define grid
cube_grid <- grid_regular(cube_params, levels = 5)

# workflow ----
cube_workflow <- workflow() %>%
  add_model(cube_model) %>%
  add_recipe(recipe)

# Tuning/fitting ----
cube_tuned <- cube_workflow %>%
  tune_grid(
    resamples = folds,
    grid = cube_grid,
    control = ctrl_grid)

write_rds(cube_tuned, "model selection/model_objects/cube_tuned.rds")


cube_workflow_tuned <- cube_workflow %>%
  finalize_workflow(select_best(cube_tuned, metric = "rmse"))



cube_fit_folds <- fit_resamples(
  cube_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T)
)



cube_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )


total_model_results <- (cube_fit_folds %>% collect_metrics() %>% mutate(model = "Cube"))

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


peek <- recipe_pca %>% prep() %>% bake(pca_train)


cube_model_pca <- parsnip::cubist_rules(
  committees = tune(),
  neighbors = tune()
)



# set-up tuning grid ----
cube_params_pca <- hardhat::extract_parameter_set_dials(cube_model_pca) %>%
  update(neighbors = neighbors(c(5,40)),
         committees = committees(c(3,40))
  )

# define grid
cube_grid_pca <- grid_regular(cube_params_pca, levels = 5)

# workflow ----
cube_workflow_pca <- workflow() %>%
  add_model(cube_model_pca) %>%
  add_recipe(recipe_pca)

# Tuning/fitting ----
cube_tuned_pca <- cube_workflow_pca %>%
  tune_grid(
    resamples = folds_pca,
    grid = cube_grid_pca,
    control = ctrl_grid)

write_rds(cube_tuned_pca, "model selection/model_objects/cube_tuned_pca.rds")

cube_workflow_tuned_pca <- cube_workflow_pca %>%
  finalize_workflow(select_best(cube_tuned_pca, metric = "rmse"))



cube_fit_folds_pca <- fit_resamples(
  cube_workflow_tuned_pca, 
  resamples = folds_pca,
  control = control_grid(save_pred = T)
)


cube_fit_folds_pca %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )


total_model_results_pca <- (cube_fit_folds_pca %>% collect_metrics() %>% mutate(model = "Cube"))

total_model_results_pca %>% 
  filter(.metric == "rmse") %>% 
  arrange(mean) 






