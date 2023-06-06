# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)
library(doMC)
library(rules)
library(baguette)
registerDoMC(detectCores())
getDoParWorkers()

set.seed(3013)



bmlp_tuned <- read_rds('model selection/model_objects/bmlp_tuned.rds')
folds <- read_rds('model selection/model_objects/folds.rds')

bmlp_workflow_tuned <- bmlp_workflow %>%
  finalize_workflow(select_best(bmlp_tuned, metric = "rmse"))



bmlp_fit_folds <- fit_resamples(
  bmlp_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T),
  
)






residual_data <- left_join(
  bmlp_fit_folds %>%
    collect_predictions() %>%
    select(.row, .pred),
  train %>%
    mutate(.row = 1:n())) %>%
  select(-.row) %>%
  mutate(
    .pred = exp(.pred),
    y = exp(y),
    residual = .pred - y
  ) 

# 
# residual_data <-as_tibble(bmlp_fit_folds %>%
#   collect_predictions() %>%
#   mutate(.pred = exp(.pred),
#          y = exp(y),
#          residual = .pred - y) %>%
#   filter(id == "Repeat1", id2 == "Fold1") %>%
#   select(-id,-id2, -.row, -.config)  %>%
#   cbind(folds$splits[[1]] %>%
#           assessment() %>%
#           select(-y)))

set.seed(3013)
residual_initial_split <- rsample::initial_split(residual_data, prop = .8, strata = residual)
train_residual <- training(residual_initial_split)
test_residual <- testing(residual_initial_split)
folds_residual <- vfold_cv(train_residual, v = 5,repeats = 3)



recipe_residual <- recipe(residual ~ .,
                          train_residual) %>%
  step_rm(id,y) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) 



peek_residual <- recipe_residual %>% prep() %>% bake(train_residual)



# Single Layer Neural Network ----
nn_model_res <- mlp(
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune(),
  mode = 'regression'
)

nn_params_res <- hardhat::extract_parameter_set_dials(nn_model_res)

nn_grid_res <- grid_regular(nn_params_res, levels = 5)

nn_workflow_res <- workflow() %>%
  add_model(nn_model_res) %>%
  add_recipe(recipe_residual)

nn_tuned_res <- nn_workflow_res %>%
  tune_grid(resamples = folds_residual,
            grid = nn_grid_res,
            control = ctrl_grid)


#write_rds(nn_tuned, "model selection/model_objects/nn_tuned.rds")


nn_workflow_tuned_res <- nn_workflow_res %>%
  finalize_workflow(select_best(nn_tuned_res, metric = "rmse"))



nn_fit_folds_res <- fit_resamples(nn_workflow_tuned_res,
                              resamples = folds_residual,
                              control = control_grid(save_pred = T))




nn_fit_folds_res %>%
  collect_predictions() %>%
  select(.pred, residual) %>%
  rmse(truth = residual,
       estimate = .pred)




nn_res_fit <- nn_workflow_tuned_res %>% fit(train_residual)



nn_res_test <-
  test_residual %>%
  bind_cols(predict(nn_res_fit, .)) %>%
  rename(.pred_res = .pred...28,
         .pred_val = .pred...1
  )


nn_res_test %>% mutate(.pred_new = .pred_val - .pred_res) %>% rmse(truth = y, estimate = .pred_new)

nn_res_test %>%rmse(truth = y, estimate = .pred_val)


predictions48 <- ensemble_test_pca %>% mutate(.pred = exp(.pred),y = 0) %>%
  bind_cols(predict(nn_res_fit, .)) %>%
  select(-y) %>%
  rename(y = .pred...28,
         residual = .pred...30
  ) %>%
  mutate(.pred_new = y - residual,
         .pred_new = ifelse(.pred_new >100, 100, .pred_new),
         .pred_new= ifelse(.pred_new <0, 0, .pred_new)
  ) %>%
  select(id,.pred_new) %>%
  rename(y = .pred_new)

write_csv(predictions48, 'predictions/prediction48.csv' )



