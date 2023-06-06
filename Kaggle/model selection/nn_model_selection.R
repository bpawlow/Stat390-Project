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


#---- Single Layer Neural Network
nn_model <- mlp(
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune(),
  mode = 'regression'
)

# set-up tuning grid ----
nn_params <- hardhat::extract_parameter_set_dials(nn_model)

# define grid
nn_grid <- grid_regular(nn_params, levels = 5) 

# workflow ----
nn_workflow <- workflow() %>%
  add_model(nn_model) %>%
  add_recipe(recipe)

# Tuning/fitting ----
nn_tuned <- nn_workflow %>%
  tune_grid(
    resamples = folds,
    grid = nn_grid,
    control = ctrl_grid
  )


#write_rds(nn_tuned, "model selection/model_objects/nn_tuned.rds")


nn_workflow_tuned <- nn_workflow %>%
  finalize_workflow(select_best(nn_tuned, metric = "rmse"))



nn_fit_folds <- fit_resamples(
  nn_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T)
)




nn_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y),
         .pred = round(.pred) + 2
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )




resample_round<-nn_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y),
         .pred = round(.pred) + 2
  ) 

ggplot(resample_round, aes(.pred,y)) + 
  geom_point()


total_model_results <- (nn_fit_folds %>% collect_metrics() %>% mutate(model = "Neural Network"))

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


peek_pca <- recipe_pca %>% prep() %>% bake(pca_train)


#---- Single Layer Neural Network
nn_model_pca <- mlp(
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune(),
  mode = 'regression'
)

# set-up tuning grid ----
nn_params_pca <- hardhat::extract_parameter_set_dials(nn_model_pca)

# define grid
nn_grid_pca <- grid_regular(nn_params_pca, levels = 5) 

# workflow ----
nn_workflow_pca <- workflow() %>%
  add_model(nn_model_pca) %>%
  add_recipe(recipe_pca)

# Tuning/fitting ----
nn_tuned_pca <- nn_workflow_pca %>%
  tune_grid(
    resamples = folds_pca,
    grid = nn_grid_pca,
    control = ctrl_grid
  )

write_rds(nn_tuned_pca, "model selection/model_objects/nn_tuned_pca.rds")


nn_workflow_tuned_pca <- nn_workflow_pca %>%
  finalize_workflow(select_best(nn_tuned_pca, metric = "rmse"))



nn_fit_folds_pca <- fit_resamples(
  nn_workflow_tuned_pca, 
  resamples = folds_pca,
  control = control_grid(save_pred = T)
)




nn_fit_folds_pca %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )




total_model_results_pca <- (nn_fit_folds_pca %>% collect_metrics() %>% mutate(model = "Neural Network"))

total_model_results_pca %>% 
  filter(.metric == "rmse")






