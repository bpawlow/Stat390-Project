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




rf_model <- rand_forest(
  mode = "regression",
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("ranger")



# set-up tuning grid ----
rf_params <- hardhat::extract_parameter_set_dials(rf_model)%>% 
  update(mtry = mtry(c(1, 22))) 

# define grid
rf_grid <- grid_regular(rf_params, levels = 5)

# workflow ----
rf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(recipe)

# Tuning/fitting ----
rf_tuned <- rf_workflow %>%
  tune_grid(
    resamples = folds,
    grid = rf_grid,
    control = ctrl_grid
    )

write_rds(rf_tuned, "model selection/model_objects/rf_tuned.rds")


rf_workflow_tuned <- rf_workflow %>%
  finalize_workflow(select_best(rf_tuned, metric = "rmse"))


rf_fit_folds <- fit_resamples(
  rf_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T)
  )



rf_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )



rf_fit_folds %>%
  collect_metrics() %>% 
  mutate(model = "Random Forests") %>%
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




rf_model_pca <- rand_forest(
  mode = "regression",
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("ranger")



# set-up tuning grid ----
rf_params_pca <- hardhat::extract_parameter_set_dials(rf_model_pca)%>% 
  update(mtry = mtry(c(1, 22))) 

# define grid
rf_grid_pca <- grid_regular(rf_params_pca, levels = 5)

# workflow ----
rf_workflow_pca <- workflow() %>%
  add_model(rf_model_pca) %>%
  add_recipe(recipe_pca)

# Tuning/fitting ----
rf_tuned_pca <- rf_workflow %>%
  tune_grid(
    resamples = folds_pca,
    grid = rf_grid_pca,
    control = ctrl_grid
    )

write_csv(rf_tuned_pca, 'model selection/ model_objects/rf_tuned_pca.rds')

rf_workflow_tuned_pca <- rf_workflow_pca %>%
  finalize_workflow(select_best(rf_tuned_pca, metric = "rmse"))


rf_fit_folds_pca <- fit_resamples(
  rf_workflow_tuned_pca, 
  resamples = folds_pca,
  control = control_grid(save_pred = T)
)



rf_fit_folds_pca %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )



rf_fit_folds_pca %>%
  collect_metrics() %>%
  mutate(model = "Random Forests") %>%
  filter(.metric == "rmse") %>%
  arrange(mean)
