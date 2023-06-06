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



# Define model ----
en_model_residual <- linear_reg(
  mixture = tune(),
  penalty = tune(),
  mode = "regression",
) %>%
  set_engine("glmnet")

en_params_residual <- hardhat::extract_parameter_set_dials(en_model_residual)

# define grid
en_grid_residual <- grid_regular(en_params_residual, levels = 5)

# workflow ----
en_workflow_residual <- workflow() %>%
  add_model(en_model_residual) %>%
  add_recipe(recipe_residual)

# Tuning/fitting ----
en_tuned_residual <- en_workflow_residual %>%
  tune_grid(
    resamples = folds_residual,
    grid = en_grid_residual,
    control = ctrl_grid)




en_workflow_tuned_residual <- en_workflow_residual %>%
  finalize_workflow(select_best(en_tuned_residual, metric = "rmse"))



en_fit_folds_residual <- fit_resamples(
  en_workflow_tuned_residual, 
  resamples = folds_residual, 
  control = control_grid(save_pred = T))



en_fit_folds_residual %>% 
  collect_predictions() %>% 
  select(.pred, residual) %>%
  rmse(truth = residual,
       estimate = .pred
  )


en_res_fit <- en_workflow_tuned_residual %>% fit(train_residual)



en_res_test <-
  test_residual %>%
  bind_cols(predict(en_res_fit, .)) %>%
  rename(.pred_res = .pred...28,
         .pred_val = .pred...1
         )


en_res_test %>% mutate(.pred_new = .pred_val - .pred_res) %>% rmse(truth = y, estimate = .pred_new)

en_res_test %>%rmse(truth = y, estimate = .pred_val)


predictions46<- ensemble_test_pca %>% mutate(.pred = exp(.pred),y = 0) %>%
  bind_cols(predict(en_res_fit, .)) %>%
  select(-y) %>%
  rename(y = .pred...28,
         residual = .pred...30
         ) %>%
  mutate(.pred_new = y - residual,
         .pred_new = ifelse(.pred_new >100, 100, .pred_new),
         .pred_new= ifelse(.pred_new <0, 0, .pred_new)
         ) %>%
  select(id,.pred_new) %>%
  rename(y = .pred_new) %>%
  mutate(y = y - 1)%>%
  mutate(
         y = ifelse(y <0, 0,y)
  ) 
  
write_csv(predictions46, 'predictions/prediction46.csv' )



